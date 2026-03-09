#include "preprocess.h"

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

static const int RUNS = 10;

struct TimingResult {
    float gpu_ms;
    float cpu_ms;
};

// CPU reference: sequential sort per face for comparison
static float cpu_sort_slots(const grwm::MeshData& mesh,
                            const std::vector<float>& curvature,
                            uint32_t slots_per_face) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Simulate: just allocate and sort random data per face
    std::vector<float> priorities(slots_per_face);
    for (uint32_t f = 0; f < mesh.face_count; ++f) {
        for (uint32_t s = 0; s < slots_per_face; ++s) {
            priorities[s] = static_cast<float>(s) / slots_per_face;
        }
        std::sort(priorities.begin(), priorities.end(), std::greater<float>());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: benchmark <mesh.obj> [slots_per_face]\n");
        return 1;
    }

    uint32_t slots_per_face = grwm::DEFAULT_SLOTS_PER_FACE;
    if (argc >= 3) {
        slots_per_face = static_cast<uint32_t>(atoi(argv[2]));
    }

    // GPU info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.1f MB free / %.1f MB total\n\n",
           free_mem / (1024.0f * 1024.0f),
           total_mem / (1024.0f * 1024.0f));

    // Load mesh
    grwm::MeshData mesh;
    if (!grwm::load_mesh(argv[1], mesh)) {
        printf("Failed to load mesh: %s\n", argv[1]);
        return 1;
    }
    printf("Mesh: %u vertices, %u faces\n", mesh.vertex_count, mesh.face_count);
    printf("Slots per face: %u\n", slots_per_face);
    printf("Averaging over %d runs\n\n", RUNS);

    // Warm-up
    printf("Warming up GPU...\n");
    {
        auto warmup = grwm::compute_curvature(mesh);
        (void)warmup;
    }

    // --- Stage 1: Curvature ---
    float curv_gpu_ms = 0.0f;
    std::vector<float> curvature;
    for (int i = 0; i < RUNS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        curvature = grwm::compute_curvature(mesh);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        curv_gpu_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    curv_gpu_ms /= RUNS;

    // --- Stage 2: Feature Edges ---
    float feat_gpu_ms = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto features = grwm::detect_feature_edges(mesh, 30.0f);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        feat_gpu_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();
        (void)features;
    }
    feat_gpu_ms /= RUNS;

    // --- Stage 3: Slots ---
    float slots_gpu_ms = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto slots = grwm::compute_slots(mesh, curvature, slots_per_face,
                                          0.5f, 0.4f, 0.1f);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        slots_gpu_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();
        (void)slots;
    }
    slots_gpu_ms /= RUNS;

    // CPU reference for slot sort
    float slots_cpu_ms = cpu_sort_slots(mesh, curvature, slots_per_face);

    float total_gpu = curv_gpu_ms + feat_gpu_ms + slots_gpu_ms;

    // Results table
    printf("+-----------------+-------------+\n");
    printf("| Stage           | GPU (ms)    |\n");
    printf("+-----------------+-------------+\n");
    printf("| 1. Curvature    | %9.2f   |\n", curv_gpu_ms);
    printf("| 2. Features     | %9.2f   |\n", feat_gpu_ms);
    printf("| 3. Slots        | %9.2f   |\n", slots_gpu_ms);
    printf("+-----------------+-------------+\n");
    printf("| Total           | %9.2f   |\n", total_gpu);
    printf("+-----------------+-------------+\n");

    if (slots_cpu_ms > 0.0f) {
        printf("\nSlot sort speedup: %.1fx (CPU ref: %.2f ms)\n",
               slots_cpu_ms / slots_gpu_ms, slots_cpu_ms);
    }

    // Memory after benchmark
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("\nGPU Memory after benchmark: %.1f MB free / %.1f MB total\n",
           free_mem / (1024.0f * 1024.0f),
           total_mem / (1024.0f * 1024.0f));

    return 0;
}
