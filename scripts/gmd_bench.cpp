/*
 * gmd_bench.cpp  —  GMD-YOLO TFLite deployment benchmark for VOXL2
 *
 * Compile on-board (see Makefile or the g++ command at bottom of this file).
 * Requires: libtensorflow-lite.so  voxl-opencv  libc (dlfcn, glob)
 *
 * Delegates loaded via dlopen — no delegate headers needed at compile time.
 * Hard-fails loudly if delegate load or ModifyGraphWithDelegate fails.
 * No silent CPU fallback.
 *
 * Usage:
 *   gmd_bench <model.tflite> <delegate> <rgb_dir> <nir_dir> [options]
 *
 *   delegate: cpu | gpu | hexagon | nnapi
 *
 * Options:
 *   --variant NAME        row label (default: delegate type)
 *   --warmup N            warmup inferences, discarded  (default 10)
 *   --burst N             timed burst inferences        (default 50)
 *   --sustained S         sustained run seconds         (default 120)
 *   --threads N           CPU threads (cpu only)        (default 4)
 *   --temp-interval S     SoC temp sample interval (s)  (default 5)
 *   --delegate-so PATH    override default .so path
 *   --out PATH            JSON output path              (default bench_result.json)
 *
 * Default .so paths:
 *   gpu     → /usr/lib/libgpu_delegate.so
 *   hexagon → /usr/lib/libhexagon_interface.so
 *   nnapi   → /usr/lib/libnnapi_delegate.so
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <dirent.h>
#include <dlfcn.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>

/* Steady-clock timing — avoid chrono::system_clock drift */
#include <time.h>

/* TFLite 2.8.0 C++ API headers */
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
/* kTfLiteFloat32, kTfLiteInt8, etc. come from interpreter.h via c_api_types.h */

/* OpenCV (voxl-opencv, typically 4.x) */
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/* ─────────────────────────────────────────────────────────────────────────── */
/* Delegate function-pointer types                                              */
/* All delegates expose a C-API returning TfLiteDelegate* (opaque struct).     */
/* ─────────────────────────────────────────────────────────────────────────── */

typedef TfLiteDelegate* (*fn_gpu_create)(const void* options);
typedef void            (*fn_gpu_delete)(TfLiteDelegate*);

/* Mirrors TfLiteHexagonDelegateOptions in TFLite 2.8.0 exactly.
   If the board's libhexagon_interface.so uses a different layout, set
   debug_level=0 and everything else 0 — the struct is POD and zero-init
   uses defaults that match the header's comments. */
struct HexagonOptions {
    int     debug_level              = 0;
    int     powersave_level          = 0;
    bool    print_graph_profile      = false;
    bool    print_graph_debug        = false;
    int     max_delegated_partitions = 10;
    int     min_nodes_per_partition  = 2;
    bool    enable_dynamic_batch_size = false;
    uint8_t input_batch_size         = 1;
};
typedef TfLiteDelegate* (*fn_hex_create)(const HexagonOptions*);
typedef void            (*fn_hex_delete)(TfLiteDelegate*);
typedef void            (*fn_hex_init)();
typedef void            (*fn_hex_teardown)();

/* NNAPI C-API (present in TFLite 2.7+, libnnapi_delegate.so) */
typedef TfLiteDelegate* (*fn_nnapi_create)(const void* options);
typedef void            (*fn_nnapi_delete)(TfLiteDelegate*);

/* ─────────────────────────────────────────────────────────────────────────── */
/* Delegate lifetime handle                                                     */
/* ─────────────────────────────────────────────────────────────────────────── */

struct DelegateCtx {
    void*           dl_handle   = nullptr;
    TfLiteDelegate* delegate    = nullptr;
    fn_gpu_delete   del_gpu     = nullptr;
    fn_hex_delete   del_hex     = nullptr;
    fn_hex_teardown hex_teardown= nullptr;
    fn_nnapi_delete del_nnapi   = nullptr;
    std::string     applied;   /* description printed in output */

    void destroy() {
        if (delegate) {
            if (del_gpu)   del_gpu(delegate);
            if (del_hex) { del_hex(delegate); if (hex_teardown) hex_teardown(); }
            if (del_nnapi) del_nnapi(delegate);
            delegate = nullptr;
        }
        if (dl_handle) { dlclose(dl_handle); dl_handle = nullptr; }
    }
};

/* ─────────────────────────────────────────────────────────────────────────── */
/* Fatal error: print and exit(1)                                              */
/* ─────────────────────────────────────────────────────────────────────────── */

static void fatal(const std::string& msg) {
    std::cerr << "\n[HARD FAIL] " << msg << "\n" << std::flush;
    exit(1);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Monotonic clock (ns, wraps steady_clock without chrono overhead)            */
/* ─────────────────────────────────────────────────────────────────────────── */

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* SoC temperature: max across all thermal zones (/sys/class/thermal)          */
/* ─────────────────────────────────────────────────────────────────────────── */

static double read_max_temp() {
    double mx = -1.0;
    glob_t g;
    if (glob("/sys/class/thermal/thermal_zone*/temp", GLOB_NOSORT, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; ++i) {
            std::ifstream f(g.gl_pathv[i]);
            int v = 0;
            if (f >> v) mx = std::max(mx, v / 1000.0);
        }
        globfree(&g);
    }
    return mx;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Statistics                                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */

static double vmean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}
static double vstd(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = vmean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}
/* Percentile on a copy (sorts in place) */
static double vpct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    int lo = (int)idx, hi = lo + 1;
    if (hi >= (int)v.size()) return v.back();
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Image loading: letterbox BGR+NIR → float32 NHWC [0,1]                      */
/* Matches dataset.py: np.concatenate([bgr, nir[:,:,None]], axis=2) / 255     */
/* ─────────────────────────────────────────────────────────────────────────── */

static bool load_4ch(const std::string& rgb_path, const std::string& nir_path,
                     float* dst, int sz)
{
    cv::Mat bgr = cv::imread(rgb_path,  cv::IMREAD_COLOR);
    cv::Mat nir = cv::imread(nir_path,  cv::IMREAD_GRAYSCALE);
    if (bgr.empty() || nir.empty()) return false;
    if (nir.size() != bgr.size())
        cv::resize(nir, nir, bgr.size(), 0, 0, cv::INTER_LINEAR);

    /* Stack into 4-channel [B,G,R,NIR] uint8 */
    std::vector<cv::Mat> chs;
    cv::split(bgr, chs);   /* chs = {B, G, R} */
    chs.push_back(nir);    /* chs = {B, G, R, NIR} */
    cv::Mat img4;
    cv::merge(chs, img4);  /* [H,W,4] uint8 */

    /* Letterbox to sz×sz with 114-gray padding */
    int h0 = img4.rows, w0 = img4.cols;
    double r  = (double)sz / std::max(h0, w0);
    int nh = (int)std::round(h0 * r), nw = (int)std::round(w0 * r);
    cv::resize(img4, img4, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    int dh = sz - nh, dw = sz - nw;
    cv::copyMakeBorder(img4, img4,
                       dh/2, dh - dh/2,
                       dw/2, dw - dw/2,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114, 114));

    /* Copy to float [H,W,4] / 255 → NHWC layout required by TFLite */
    for (int h = 0; h < sz; ++h) {
        const uint8_t* row = img4.ptr<uint8_t>(h);
        float*         drow = dst + h * sz * 4;
        for (int w = 0; w < sz; ++w) {
            drow[w*4+0] = row[w*4+0] * (1.f/255.f);   /* B   */
            drow[w*4+1] = row[w*4+1] * (1.f/255.f);   /* G   */
            drow[w*4+2] = row[w*4+2] * (1.f/255.f);   /* R   */
            drow[w*4+3] = row[w*4+3] * (1.f/255.f);   /* NIR */
        }
    }
    return true;
}

/* Scan rgb_dir/*.jpg and match NIR by stem */
static std::vector<std::pair<std::string,std::string>>
scan_pairs(const std::string& rgb_dir, const std::string& nir_dir)
{
    std::vector<std::pair<std::string,std::string>> pairs;
    glob_t g;
    std::string pat = rgb_dir + "/*.jpg";
    if (glob(pat.c_str(), GLOB_NOSORT, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; ++i) {
            std::string rp = g.gl_pathv[i];
            std::string fname = rp.substr(rp.rfind('/') + 1);
            std::string stem  = fname.substr(0, fname.rfind('.'));
            std::string np    = nir_dir + "/" + stem + ".png";
            if (access(np.c_str(), R_OK) == 0)
                pairs.push_back(std::make_pair(rp, np));
        }
        globfree(&g);
    }
    return pairs;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Delegate loaders — each calls fatal() on any failure                        */
/* ─────────────────────────────────────────────────────────────────────────── */

static DelegateCtx load_gpu(const std::string& so_path) {
    DelegateCtx ctx;
    ctx.dl_handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!ctx.dl_handle)
        fatal("dlopen GPU delegate '" + so_path + "' failed:\n  " + dlerror());

    fn_gpu_create create =
        (fn_gpu_create)dlsym(ctx.dl_handle, "TfLiteGpuDelegateV2Create");
    if (!create)
        fatal("Symbol TfLiteGpuDelegateV2Create not in " + so_path +
              "\n  (may need a different .so — check with: nm -D " + so_path + " | grep Gpu)");

    ctx.del_gpu = (fn_gpu_delete)dlsym(ctx.dl_handle, "TfLiteGpuDelegateV2Delete");

    /* nullptr → TfLiteGpuDelegateOptionsV2Default() equivalent */
    ctx.delegate = create(nullptr);
    if (!ctx.delegate)
        fatal("TfLiteGpuDelegateV2Create returned null — Adreno GPU not accessible?");

    ctx.applied = "GPU (" + so_path + ")";
    return ctx;
}

static DelegateCtx load_hexagon(const std::string& so_path) {
    DelegateCtx ctx;
    ctx.dl_handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!ctx.dl_handle)
        fatal("dlopen Hexagon delegate '" + so_path + "' failed:\n  " + dlerror());

    fn_hex_init init_fn =
        (fn_hex_init)dlsym(ctx.dl_handle, "TfLiteHexagonInit");
    if (!init_fn)
        fatal("Symbol TfLiteHexagonInit not in " + so_path +
              "\n  (check: nm -D " + so_path + " | grep Hexagon)");

    fn_hex_create create =
        (fn_hex_create)dlsym(ctx.dl_handle, "TfLiteHexagonDelegateCreate");
    if (!create)
        fatal("Symbol TfLiteHexagonDelegateCreate not in " + so_path);

    ctx.hex_teardown = (fn_hex_teardown)dlsym(ctx.dl_handle, "TfLiteHexagonTearDown");
    ctx.del_hex      = (fn_hex_delete)  dlsym(ctx.dl_handle, "TfLiteHexagonDelegateDelete");

    /* TfLiteHexagonInit() locates libhexagon_nn_stub.so / libhexagon_nn_skel.so.
       It will print DSP initialization messages to stderr. */
    init_fn();

    HexagonOptions opts;   /* zero-initialized, matches TFLite 2.8.0 defaults */
    ctx.delegate = create(&opts);
    if (!ctx.delegate)
        fatal("TfLiteHexagonDelegateCreate returned null.\n"
              "  Check: libhexagon_nn_skel.so present? DSP accessible?\n"
              "  Run: adb logcat | grep -i hexagon");

    ctx.applied = "Hexagon/NPU (" + so_path + ")";
    return ctx;
}

static DelegateCtx load_nnapi(const std::string& so_path) {
    DelegateCtx ctx;
    ctx.dl_handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!ctx.dl_handle)
        fatal("dlopen NNAPI delegate '" + so_path + "' failed:\n  " + dlerror());

    fn_nnapi_create create =
        (fn_nnapi_create)dlsym(ctx.dl_handle, "TfLiteNnapiDelegateCreate");
    if (!create)
        fatal("Symbol TfLiteNnapiDelegateCreate not in " + so_path +
              "\n  (NNAPI C-API may be absent in this build; try hexagon instead)");

    ctx.del_nnapi = (fn_nnapi_delete)dlsym(ctx.dl_handle, "TfLiteNnapiDelegateDelete");

    ctx.delegate = create(nullptr);   /* nullptr → default options */
    if (!ctx.delegate)
        fatal("TfLiteNnapiDelegateCreate returned null.\n"
              "  NNAPI HAL is an Android concept; may not be available on Linux/VOXL2.\n"
              "  Use --delegate hexagon instead for DSP acceleration.");

    ctx.applied = "NNAPI (" + so_path + ")";
    return ctx;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Minimal JSON builder (no external library)                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static std::string jq(const std::string& s) { return "\"" + s + "\""; }
static std::string jd(double v, int p = 2) {
    std::ostringstream o; o.precision(p); o << std::fixed << v; return o.str();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Usage                                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <model.tflite> <delegate> <rgb_dir> <nir_dir> [options]\n"
        "  delegate: cpu | gpu | hexagon | nnapi\n"
        "Options:\n"
        "  --variant NAME        table row label\n"
        "  --warmup N            warmup runs, discarded  (default 10)\n"
        "  --burst N             timed burst runs        (default 50)\n"
        "  --sustained S         sustained run seconds   (default 120)\n"
        "  --threads N           CPU threads (cpu only)  (default 4)\n"
        "  --temp-interval S     SoC temp sample (s)     (default 5)\n"
        "  --delegate-so PATH    override delegate .so\n"
        "  --out PATH            JSON output             (default bench_result.json)\n"
        "Default .so: gpu=/usr/lib/libgpu_delegate.so\n"
        "             hexagon=/usr/lib/libhexagon_interface.so\n"
        "             nnapi=/usr/lib/libnnapi_delegate.so\n",
        prog);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* main                                                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

int main(int argc, char* argv[]) {
    if (argc < 5) { usage(argv[0]); return 1; }

    std::string model_path = argv[1];
    std::string del_type   = argv[2];
    std::string rgb_dir    = argv[3];
    std::string nir_dir    = argv[4];

    /* Defaults */
    int         warmup      = 10;
    int         burst_n     = 50;
    int         sustained_s = 120;
    int         cpu_threads = 4;
    double      temp_ivl_s  = 5.0;
    std::string del_so;
    std::string out_json    = "bench_result.json";
    std::string variant;

    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--warmup"       && i+1<argc) warmup       = std::stoi(argv[++i]);
        else if (a == "--burst"        && i+1<argc) burst_n      = std::stoi(argv[++i]);
        else if (a == "--sustained"    && i+1<argc) sustained_s  = std::stoi(argv[++i]);
        else if (a == "--threads"      && i+1<argc) cpu_threads  = std::stoi(argv[++i]);
        else if (a == "--temp-interval"&& i+1<argc) temp_ivl_s   = std::stod(argv[++i]);
        else if (a == "--delegate-so"  && i+1<argc) del_so       = argv[++i];
        else if (a == "--out"          && i+1<argc) out_json     = argv[++i];
        else if (a == "--variant"      && i+1<argc) variant      = argv[++i];
    }
    if (variant.empty()) variant = del_type;

    /* Default .so paths */
    if (del_so.empty()) {
        if      (del_type == "gpu")     del_so = "/usr/lib/libgpu_delegate.so";
        else if (del_type == "hexagon") del_so = "/usr/lib/libhexagon_interface.so";
        else if (del_type == "nnapi")   del_so = "/usr/lib/libnnapi_delegate.so";
    }

    printf("================================================================\n");
    printf("GMD-YOLO VOXL2 Benchmark  —  variant: %s\n", variant.c_str());
    printf("  model    : %s\n", model_path.c_str());
    printf("  delegate : %s", del_type.c_str());
    if (!del_so.empty()) printf("  (%s)", del_so.c_str());
    printf("\n  warmup=%d  burst=%d  sustained=%ds  threads=%d\n",
           warmup, burst_n, sustained_s, cpu_threads);
    printf("================================================================\n");

    /* ── Load model ───────────────────────────────────────────────────────── */
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) fatal("Cannot load model: " + model_path);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp;
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&interp) != kTfLiteOk || !interp)
        fatal("InterpreterBuilder failed");
    printf("[ok] Model loaded\n");

    /* ── Load delegate ────────────────────────────────────────────────────── */
    DelegateCtx dctx;
    if (del_type == "cpu") {
        interp->SetNumThreads(cpu_threads);
        dctx.applied = "CPU (" + std::to_string(cpu_threads) + " threads)";
    } else if (del_type == "gpu") {
        dctx = load_gpu(del_so);
    } else if (del_type == "hexagon") {
        dctx = load_hexagon(del_so);
    } else if (del_type == "nnapi") {
        dctx = load_nnapi(del_so);
    } else {
        fatal("Unknown delegate '" + del_type + "'. Use: cpu | gpu | hexagon | nnapi");
    }

    if (dctx.delegate) {
        TfLiteStatus s = interp->ModifyGraphWithDelegate(dctx.delegate);
        if (s != kTfLiteOk) {
            dctx.destroy();
            fatal("ModifyGraphWithDelegate(" + del_type + ") failed.\n"
                  "  Check: model dtype compatible with delegate?\n"
                  "  Hexagon requires INT8 weights. GPU works with float32/float16.");
        }
        printf("[ok] Delegate applied: %s\n", dctx.applied.c_str());
    } else {
        printf("[ok] Delegate: %s\n", dctx.applied.c_str());
    }

    if (interp->AllocateTensors() != kTfLiteOk) {
        dctx.destroy();
        fatal("AllocateTensors() failed.\n"
              "  For FP16/CPU: ARM FP16 NEON required (should work on QRB5165N).\n"
              "  For INT8/NPU: Hexagon may not support all ops — check logcat.");
    }

    /* ── Validate input tensor ────────────────────────────────────────────── */
    TfLiteTensor* in_t = interp->input_tensor(0);
    if (!in_t) fatal("No input tensor");
    if (in_t->type != kTfLiteFloat32)
        fatal("Expected kTfLiteFloat32 input (type 1), got type " +
              std::to_string(in_t->type) +
              ".\n  The INT8-mixed model has float32 I/O — check you're using the right model.");

    int H = in_t->dims->data[1];
    int W = in_t->dims->data[2];
    int C = in_t->dims->data[3];
    printf("[ok] Input: [1,%d,%d,%d]  float32\n", H, W, C);
    if (C != 4) fatal("Expected 4-channel input (BGRN), got " + std::to_string(C));

    float* in_ptr = in_t->data.f;  /* direct pointer into TFLite arena */
    if (!in_ptr) fatal("Input tensor data pointer is null after AllocateTensors");

    TfLiteTensor* out_t = interp->output_tensor(0);
    printf("[ok] Output: [1,%d,%d]  dtype=%d\n",
           out_t->dims->data[1], out_t->dims->data[2], (int)out_t->type);

    /* ── Load images ──────────────────────────────────────────────────────── */
    auto pairs = scan_pairs(rgb_dir, nir_dir);
    if (pairs.empty())
        fatal("No RGB/NIR pairs found.\n"
              "  RGB dir: " + rgb_dir + "/*.jpg\n"
              "  NIR dir: " + nir_dir + "/*.png\n"
              "  Push images first: see adb push commands.");
    printf("[ok] Found %zu image pair(s)\n", pairs.size());

    /* Preload up to 30 images as float32 letterboxed — cycles during sustained run.
       30 × 1280 × 1280 × 4 × 4 bytes = 786 MB.  If OOM, reduce to 10. */
    int n_pre = std::min((int)pairs.size(), 30);
    size_t img_floats = (size_t)H * W * C;
    std::vector<std::vector<float>> imgs(n_pre, std::vector<float>(img_floats));
    int n_loaded = 0;
    for (int i = 0; i < n_pre; ++i) {
        if (load_4ch(pairs[i].first, pairs[i].second, imgs[i].data(), H))
            ++n_loaded;
        else
            fprintf(stderr, "  WARN: cannot load %s\n", pairs[i].first.c_str());
    }
    if (n_loaded == 0) fatal("No images loaded successfully — check paths and OpenCV build.");
    printf("[ok] Preloaded %d / %d images\n", n_loaded, n_pre);

    /* ── Warmup ───────────────────────────────────────────────────────────── */
    printf("\nWarmup (%d)...\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        std::memcpy(in_ptr, imgs[i % n_loaded].data(), img_floats * sizeof(float));
        if (interp->Invoke() != kTfLiteOk) fatal("Invoke() failed during warmup");
    }
    printf("[ok] Warmup done\n");

    /* ── Burst (latency percentiles + FPS) ───────────────────────────────── */
    printf("\nBurst (%d inferences)...\n", burst_n);
    std::vector<double> lat_ms;
    lat_ms.reserve(burst_n);
    bool burst_crash = false;
    for (int i = 0; i < burst_n; ++i) {
        std::memcpy(in_ptr, imgs[i % n_loaded].data(), img_floats * sizeof(float));
        double t0 = now_ms();
        TfLiteStatus st = interp->Invoke();
        lat_ms.push_back(now_ms() - t0);
        if (st != kTfLiteOk) { burst_crash = true; break; }
    }
    if (burst_crash) fprintf(stderr, "[CRASH] Invoke() failed during burst\n");

    double lmean = vmean(lat_ms);
    double lstd  = vstd(lat_ms);
    double lp50  = vpct(lat_ms, 50.0);
    double lp95  = vpct(lat_ms, 95.0);
    double lp99  = vpct(lat_ms, 99.0);
    double fps_b = (lmean > 0) ? 1000.0 / lmean : 0.0;

    printf("  mean=%.1f ms  std=%.1f ms\n", lmean, lstd);
    printf("  p50=%.1f  p95=%.1f  p99=%.1f  (ms)\n", lp50, lp95, lp99);
    printf("  FPS_burst=%.2f\n", fps_b);

    double temp_after_burst = read_max_temp();
    if (temp_after_burst > 0)
        printf("  SoC temp (burst end): %.1f°C\n", temp_after_burst);

    /* ── Sustained run ────────────────────────────────────────────────────── */
    printf("\nSustained run (%d s)...\n", sustained_s);
    std::vector<std::pair<double,double>> temp_trace; /* {t_s, max_temp_c} */
    int    sus_frames = 0;
    bool   sus_crash  = false;
    double t0_sus      = now_ms();
    double t_last_temp = t0_sus;
    double t_last_log  = t0_sus;

    while (true) {
        double tnow    = now_ms();
        double elapsed = tnow - t0_sus;
        if (elapsed >= sustained_s * 1000.0) break;

        std::memcpy(in_ptr, imgs[sus_frames % n_loaded].data(), img_floats * sizeof(float));
        if (interp->Invoke() != kTfLiteOk) { sus_crash = true; break; }
        ++sus_frames;

        /* Temperature sample */
        if ((tnow - t_last_temp) >= temp_ivl_s * 1000.0) {
            double tc = read_max_temp();
            temp_trace.push_back(std::make_pair(elapsed / 1000.0, tc));
            t_last_temp = tnow;
        }
        /* Progress log every 30 s */
        if ((tnow - t_last_log) >= 30000.0) {
            double fps_now = sus_frames / (elapsed / 1000.0);
            double tc = temp_trace.empty() ? -1.0 : temp_trace.back().second;
            printf("  %5.0f s  frames=%d  FPS=%.2f  temp=%.1f°C\n",
                   elapsed/1000.0, sus_frames, fps_now, tc);
            t_last_log = tnow;
        }
    }

    double sus_actual  = (now_ms() - t0_sus) / 1000.0;
    double fps_sus     = (sus_actual > 0) ? sus_frames / sus_actual : 0.0;

    double temp_max_sus = -1.0, temp_final_sus = -1.0;
    for (size_t i = 0; i < temp_trace.size(); ++i) {
        temp_max_sus   = std::max(temp_max_sus, temp_trace[i].second);
        temp_final_sus = temp_trace[i].second;
    }

    printf("  FPS_sustained=%.2f  (frames=%d  elapsed=%.1f s)\n",
           fps_sus, sus_frames, sus_actual);
    if (temp_max_sus > 0)
        printf("  SoC temp_max=%.1f°C  temp_final=%.1f°C  (%zu samples)\n",
               temp_max_sus, temp_final_sus, temp_trace.size());
    if (sus_crash)
        fprintf(stderr, "[CRASH] Invoke() failed during sustained run at frame %d\n", sus_frames);

    /* ── Table 2 row (human-readable) ────────────────────────────────────── */
    printf("\n─────────────────────────────────────────────────────────────────\n");
    printf("TABLE 2 ROW\n");
    printf("%-14s %-10s %8s %7s %7s %7s %9s %9s %8s\n",
           "variant", "delegate", "lat_mean", "p50", "p95", "p99",
           "fps_burst", "fps_sus", "temp_max");
    printf("%-14s %-10s %7.1fms %6.1fms %6.1fms %6.1fms %8.2f %8.2f",
           variant.c_str(), del_type.c_str(),
           lmean, lp50, lp95, lp99, fps_b, fps_sus);
    if (temp_max_sus > 0) printf("  %5.1f°C", temp_max_sus);
    else                   printf("  N/A");
    printf("\n─────────────────────────────────────────────────────────────────\n");

    /* ── JSON ─────────────────────────────────────────────────────────────── */
    {
        std::ostringstream j;
        j << "{\n";
        j << "  " << jq("variant")          << ": " << jq(variant)      << ",\n";
        j << "  " << jq("model")            << ": " << jq(model_path)   << ",\n";
        j << "  " << jq("delegate_type")    << ": " << jq(del_type)     << ",\n";
        j << "  " << jq("delegate_applied") << ": " << jq(dctx.applied) << ",\n";
        j << "  " << jq("imgsz")            << ": " << H                << ",\n";
        j << "  " << jq("n_images_preloaded") << ": " << n_loaded       << ",\n";
        j << "  " << jq("burst") << ": {\n"
          << "    " << jq("n")          << ": " << burst_n       << ",\n"
          << "    " << jq("lat_mean_ms")<< ": " << jd(lmean)    << ",\n"
          << "    " << jq("lat_std_ms") << ": " << jd(lstd)     << ",\n"
          << "    " << jq("p50_ms")     << ": " << jd(lp50)     << ",\n"
          << "    " << jq("p95_ms")     << ": " << jd(lp95)     << ",\n"
          << "    " << jq("p99_ms")     << ": " << jd(lp99)     << ",\n"
          << "    " << jq("fps_burst")  << ": " << jd(fps_b)    << ",\n"
          << "    " << jq("crashed")    << ": " << (burst_crash ? "true" : "false") << "\n"
          << "  },\n";
        j << "  " << jq("sustained") << ": {\n"
          << "    " << jq("secs_requested") << ": " << sustained_s          << ",\n"
          << "    " << jq("secs_actual")    << ": " << jd(sus_actual, 1)    << ",\n"
          << "    " << jq("frames")         << ": " << sus_frames           << ",\n"
          << "    " << jq("fps_sustained")  << ": " << jd(fps_sus)          << ",\n"
          << "    " << jq("crashed")        << ": " << (sus_crash ? "true" : "false") << "\n"
          << "  },\n";
        j << "  " << jq("soc_temp") << ": {\n"
          << "    " << jq("temp_max_c")  << ": "
          << (temp_max_sus > 0 ? jd(temp_max_sus, 1)   : "null") << ",\n"
          << "    " << jq("temp_final_c")<< ": "
          << (temp_final_sus > 0 ? jd(temp_final_sus, 1) : "null") << ",\n"
          << "    " << jq("n_samples")   << ": " << temp_trace.size() << ",\n"
          << "    " << jq("trace")       << ": [";
        for (size_t i = 0; i < temp_trace.size(); ++i) {
            if (i) j << ", ";
            j << "{" << jq("t_s")    << ": " << jd(temp_trace[i].first, 1)
              << ", " << jq("temp_c")<< ": " << jd(temp_trace[i].second, 1) << "}";
        }
        j << "]\n  },\n";
        /* GFLOPs is static, measured on PC — not a device metric */
        j << "  " << jq("gflops_note")
          << ": \"164.7 @ imgsz=1280, 40.8 @ imgsz=640 (PC static, not re-measured)\"\n";
        j << "}\n";

        std::ofstream of(out_json);
        if (!of)
            fprintf(stderr, "[WARN] Cannot write %s\n", out_json.c_str());
        else
            of << j.str();
    }
    printf("Results saved: %s\n", out_json.c_str());

    dctx.destroy();
    return (burst_crash || sus_crash) ? 2 : 0;
}

/*
 * ─────────────────────────────────────────────────────────────────────────────
 * COMPILE ON BOARD
 * ─────────────────────────────────────────────────────────────────────────────
 * Confirm OpenCV library name first:
 *   ls /usr/lib/libopencv_world* /usr/lib/libopencv_core* 2>/dev/null
 *
 * If libopencv_world.so exists (single-lib build):
 *   g++ -O2 -std=c++14 \
 *       -I/usr/include -I/usr/include/opencv4 \
 *       gmd_bench.cpp -o gmd_bench \
 *       -L/usr/lib \
 *       -ltensorflow-lite \
 *       -lopencv_world \
 *       -ldl -lpthread -lm \
 *       -Wl,-rpath,/usr/lib
 *
 * If split libraries (libopencv_core, etc.):
 *   g++ -O2 -std=c++14 \
 *       -I/usr/include -I/usr/include/opencv4 \
 *       gmd_bench.cpp -o gmd_bench \
 *       -L/usr/lib \
 *       -ltensorflow-lite \
 *       -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
 *       -ldl -lpthread -lm \
 *       -Wl,-rpath,/usr/lib
 *
 * If pkg-config is available (preferred):
 *   g++ -O2 -std=c++14 \
 *       -I/usr/include \
 *       $(pkg-config --cflags opencv4) \
 *       gmd_bench.cpp -o gmd_bench \
 *       -L/usr/lib -ltensorflow-lite \
 *       $(pkg-config --libs opencv4) \
 *       -ldl -lpthread -lm \
 *       -Wl,-rpath,/usr/lib
 * ─────────────────────────────────────────────────────────────────────────────
 */
