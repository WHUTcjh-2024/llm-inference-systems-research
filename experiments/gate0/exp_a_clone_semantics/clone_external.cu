// Control for clone_semantics.cu: graphs without allocation nodes can have
// multiple executable instances, but captured external pointers are shared
// until one executable's kernel parameters are explicitly rebound.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                        \
    do {                                                                   \
        const cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(error));                      \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

__global__ void work(int* workspace, unsigned long long* output,
                     int* counter, int length, long long spin) {
    const int tag = atomicAdd(counter, 1);
    const int expected = tag + 1;
    output[2 * tag] = reinterpret_cast<unsigned long long>(workspace);
    for (int i = 0; i < length; ++i) workspace[i] = expected;
    for (volatile long long i = 0; i < spin; ++i) {}
    int mismatches = 0;
    for (int i = 0; i < length; ++i) mismatches += workspace[i] != expected;
    output[2 * tag + 1] = mismatches;
}

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 20;
    long long spin = argc > 2 ? std::atoll(argv[2]) : 5000000;
    int length = argc > 3 ? std::atoi(argv[3]) : (1 << 20);
    if (repeats <= 0 || spin < 0 || length <= 0) return 2;

    const size_t bytes = static_cast<size_t>(length) * sizeof(int);
    int *workspace1 = nullptr, *workspace2 = nullptr, *counter = nullptr;
    unsigned long long* output = nullptr;
    size_t free_start = 0, total = 0;
    CHECK(cudaMemGetInfo(&free_start, &total));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace1), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace2), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&counter), sizeof(int)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&output), 4 * sizeof(*output)));
    size_t free_after_alloc = 0;
    CHECK(cudaMemGetInfo(&free_after_alloc, &total));

    cudaStream_t stream1, stream2;
    CHECK(cudaStreamCreate(&stream1));
    CHECK(cudaStreamCreate(&stream2));

    void* args1[] = {&workspace1, &output, &counter, &length, &spin};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(work);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args1;

    cudaGraph_t graph;
    cudaGraphNode_t node;
    CHECK(cudaGraphCreate(&graph, 0));
    CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    cudaGraphExec_t exec1, exec2;
    CHECK(cudaGraphInstantiate(&exec1, graph, 0));
    CHECK(cudaGraphInstantiate(&exec2, graph, 0));
    size_t free_after_instantiate = 0;
    CHECK(cudaMemGetInfo(&free_after_instantiate, &total));

    // First replay of each exec separately: a graph exec may lazily upload
    // device-side backing memory on its first launch rather than at
    // instantiate time. Attribute that cost per exec, not to the first
    // concurrent pair.
    CHECK(cudaMemset(counter, 0, sizeof(int)));
    CHECK(cudaMemset(output, 0, 4 * sizeof(*output)));
    CHECK(cudaGraphLaunch(exec1, stream1));
    CHECK(cudaStreamSynchronize(stream1));
    size_t free_after_replay1 = 0;
    CHECK(cudaMemGetInfo(&free_after_replay1, &total));
    CHECK(cudaMemset(counter, 0, sizeof(int)));
    CHECK(cudaMemset(output, 0, 4 * sizeof(*output)));
    CHECK(cudaGraphLaunch(exec2, stream2));
    CHECK(cudaStreamSynchronize(stream2));
    size_t free_after_replay2 = 0;
    CHECK(cudaMemGetInfo(&free_after_replay2, &total));

    std::fprintf(
        stderr,
        "meta: bytes_per_ws=%zu free_start=%.2fMB after_alloc=%.2fMB "
        "after_instantiate=%.2fMB after_first_replay_e1=%.2fMB "
        "after_first_replay_e2=%.2fMB "
        "delta_alloc=%.2fMB delta_instantiate=%.2fMB "
        "delta_replay_e1=%.2fMB delta_replay_e2=%.2fMB\n",
        bytes, free_start / 1048576.0, free_after_alloc / 1048576.0,
        free_after_instantiate / 1048576.0, free_after_replay1 / 1048576.0,
        free_after_replay2 / 1048576.0,
        (free_start - free_after_alloc) / 1048576.0,
        (free_after_alloc - free_after_instantiate) / 1048576.0,
        (free_after_instantiate - free_after_replay1) / 1048576.0,
        (free_after_replay1 - free_after_replay2) / 1048576.0);


    auto run = [&](const char* mode, bool concurrent) {
        for (int repeat = 0; repeat < repeats; ++repeat) {
            CHECK(cudaMemset(counter, 0, sizeof(int)));
            CHECK(cudaMemset(output, 0, 4 * sizeof(*output)));
            const auto start = std::chrono::steady_clock::now();
            CHECK(cudaGraphLaunch(exec1, stream1));
            CHECK(cudaGraphLaunch(exec2, concurrent ? stream2 : stream1));
            CHECK(cudaStreamSynchronize(stream1));
            CHECK(cudaStreamSynchronize(stream2));
            const auto stop = std::chrono::steady_clock::now();
            unsigned long long host[4];
            CHECK(cudaMemcpy(host, output, sizeof(host), cudaMemcpyDeviceToHost));
            const double milliseconds =
                std::chrono::duration<double, std::milli>(stop - start).count();
            std::printf("%s,%d,%llu,%llu,%d,%llu,%llu,%d,%.3f\n", mode, repeat,
                        host[0], host[2], host[0] == host[2], host[1], host[3],
                        host[1] != 0 || host[3] != 0, milliseconds);
        }
    };

    std::puts("mode,repeat,addr0,addr1,addr_same,bad0,bad1,race,wall_ms");
    run("shared_concurrent", true);

    void* args2[] = {&workspace2, &output, &counter, &length, &spin};
    params.kernelParams = args2;
    CHECK(cudaGraphExecKernelNodeSetParams(exec2, node, &params));
    run("separate_serial", false);
    run("separate_concurrent", true);

    CHECK(cudaGraphExecDestroy(exec2));
    CHECK(cudaGraphExecDestroy(exec1));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream2));
    CHECK(cudaStreamDestroy(stream1));
    CHECK(cudaFree(output));
    CHECK(cudaFree(counter));
    CHECK(cudaFree(workspace2));
    CHECK(cudaFree(workspace1));
    return 0;
}
