// Experiment A: do multiple instantiations of the same CUDA graph share
// graph-allocated workspace memory, and can they replay concurrently?
//
// Build (H20 / SM90):  make ARCH=-arch=sm_90
// Build (A100 / SM80): make ARCH=-arch=sm_80
// Build (RTX 4090):    make ARCH=-arch=sm_89
//
// Run:
//   ./clone_semantics --iters 1048576 --spin 50000000 --repeats 200 --mode both
//
// stdout is CSV; stderr is metadata (memory deltas, workspace size).
//
// The kernel writes its workspace pointer and a corruption count into `out`.
// Two execs are launched back-to-back; each gets a distinct tag from an
// atomic counter, writes that tag across the workspace, spins, then counts
// mismatches. If the workspace is shared and the launches overlap, both
// execs observe mismatches.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error: %s\n  at %s:%d\n",            \
                         cudaGetErrorString(err__), __FILE__, __LINE__);     \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void work(int* ws, unsigned long long* out, int* counter,
                     int iters, long long spin) {
    const int tag = atomicAdd(counter, 1);
    // Report the workspace pointer this exec actually sees.
    out[2 * tag + 0] = (unsigned long long)ws;

    const int expect = tag + 1;
    for (int i = 0; i < iters; ++i) ws[i] = expect;

    // Widen the window so concurrent launches overlap.
    for (volatile long long t = 0; t < spin; ++t) {
    }

    int bad = 0;
    for (int i = 0; i < iters; ++i) {
        if (ws[i] != expect) ++bad;
    }
    out[2 * tag + 1] = (unsigned long long)bad;
}

struct Args {
    int iters = 1 << 20;          // 4 MiB workspace
    long long spin = 50000000;    // tune to ~1 ms
    int repeats = 200;
    std::string mode = "both";    // serial | concurrent | both
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--iters" && i + 1 < argc) {
            a.iters = std::atoi(argv[++i]);
        } else if (k == "--spin" && i + 1 < argc) {
            a.spin = std::atoll(argv[++i]);
        } else if (k == "--repeats" && i + 1 < argc) {
            a.repeats = std::atoi(argv[++i]);
        } else if (k == "--mode" && i + 1 < argc) {
            a.mode = argv[++i];
        } else if (k == "--help" || k == "-h") {
            std::printf(
                "usage: %s [--iters N] [--spin N] [--repeats N] "
                "[--mode serial|concurrent|both]\n",
                argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", k.c_str());
            std::exit(2);
        }
    }
    if (a.mode != "serial" && a.mode != "concurrent" && a.mode != "both") {
        std::fprintf(stderr, "bad --mode: %s\n", a.mode.c_str());
        std::exit(2);
    }
    return a;
}

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    const size_t ws_bytes = (size_t)args.iters * sizeof(int);

    cudaStream_t s_capture, s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s_capture));
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    int* counter = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&counter), sizeof(int)));

    // Each run uses tags 0 and 1, so 4 slots are enough; keep headroom.
    constexpr int kOutSlots = 16;
    unsigned long long* out = nullptr;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&out),
                   kOutSlots * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(out, 0, kOutSlots * sizeof(unsigned long long)));

    size_t free_before = 0, free_after_e1 = 0, free_after_e2 = 0;
    size_t free_after_warmup = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

    // Capture one graph. cudaMallocAsync during capture becomes a graph
    // memory-allocation node, which is exactly the workspace we care about.
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s_capture, cudaStreamCaptureModeThreadLocal));
    int* ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ws), ws_bytes, s_capture));
    work<<<1, 1, 0, s_capture>>>(ws, out, counter, args.iters, args.spin);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(ws, s_capture));
    CUDA_CHECK(cudaStreamEndCapture(s_capture, &graph));

    // Two execs from the SAME graph. CUDA 12 uses the 3-arg form; CUDA 11
    // needs the older 5-arg form.
    cudaGraphExec_t e1 = nullptr, e2 = nullptr;
#if CUDART_VERSION >= 12000
    CUDA_CHECK(cudaGraphInstantiate(&e1, graph, 0));
    CUDA_CHECK(cudaMemGetInfo(&free_after_e1, &total));
    CUDA_CHECK(cudaGraphInstantiate(&e2, graph, 0));
#else
    CUDA_CHECK(cudaGraphInstantiate(&e1, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaMemGetInfo(&free_after_e1, &total));
    CUDA_CHECK(cudaGraphInstantiate(&e2, graph, nullptr, nullptr, 0));
#endif
    CUDA_CHECK(cudaMemGetInfo(&free_after_e2, &total));

    std::fprintf(stderr,
                 "meta: ws_bytes=%zu free_before=%.1fMB after_e1=%.1fMB "
                 "after_e2=%.1fMB delta_e1=%.1fMB delta_e2=%.1fMB\n",
                 ws_bytes, free_before / 1048576.0, free_after_e1 / 1048576.0,
                 free_after_e2 / 1048576.0,
                 (free_before - free_after_e1) / 1048576.0,
                 (free_after_e1 - free_after_e2) / 1048576.0);

    // Warm up: the graph pool may allocate lazily on first launch.
    CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
    CUDA_CHECK(cudaGraphLaunch(e1, s1));
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaMemGetInfo(&free_after_warmup, &total));
    std::fprintf(stderr, "meta: after_warmup=%.1fMB delta_warmup=%.1fMB\n",
                 free_after_warmup / 1048576.0,
                 (free_after_e2 - free_after_warmup) / 1048576.0);

    std::printf("mode,repeat,addr0,addr1,addr_same,bad0,bad1,race,wall_ms\n");

    auto run = [&](bool concurrent, int repeat) {
        CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(out, 0, 4 * sizeof(unsigned long long)));

        const auto t0 = std::chrono::steady_clock::now();
        if (concurrent) {
            CUDA_CHECK(cudaGraphLaunch(e1, s1));
            CUDA_CHECK(cudaGraphLaunch(e2, s2));
            CUDA_CHECK(cudaStreamSynchronize(s1));
            CUDA_CHECK(cudaStreamSynchronize(s2));
        } else {
            CUDA_CHECK(cudaGraphLaunch(e1, s1));
            CUDA_CHECK(cudaGraphLaunch(e2, s1));
            CUDA_CHECK(cudaStreamSynchronize(s1));
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        unsigned long long h[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));

        const bool same = (h[0] == h[2]);
        const bool race = (h[1] != 0 || h[3] != 0);
        std::printf("%s,%d,%llu,%llu,%d,%llu,%llu,%d,%.3f\n",
                    concurrent ? "concurrent" : "serial", repeat,
                    (unsigned long long)h[0], (unsigned long long)h[2],
                    same ? 1 : 0, (unsigned long long)h[1],
                    (unsigned long long)h[3], race ? 1 : 0, wall_ms);
    };

    const bool do_serial = (args.mode == "serial" || args.mode == "both");
    const bool do_concurrent =
        (args.mode == "concurrent" || args.mode == "both");

    if (do_serial) {
        for (int r = 0; r < args.repeats; ++r) run(false, r);
    }
    if (do_concurrent) {
        for (int r = 0; r < args.repeats; ++r) run(true, r);
    }

    CUDA_CHECK(cudaGraphExecDestroy(e1));
    CUDA_CHECK(cudaGraphExecDestroy(e2));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(s_capture));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUDA_CHECK(cudaFree(counter));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
