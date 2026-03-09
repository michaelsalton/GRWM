# Feature 24: Benchmark Framework

## Context

The primary performance claim of GRWM is that CUDA implementations are significantly faster than single-threaded CPU references. The benchmark framework measures and compares execution times across mesh sizes to validate these claims.

## Requirements

1. Load mesh from command-line argument
2. Time each pipeline stage separately using CUDA events (GPU) and `std::chrono` (CPU)
3. Include memory transfer time in GPU measurements
4. Average over 10 runs per stage
5. Print results in a formatted table
6. Optionally compare against CPU reference implementations

## Files Modified

- `tests/benchmark.cpp` — full implementation

## Implementation Details

### Timing Infrastructure

```cpp
// GPU timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... GPU work ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

```cpp
// CPU timing
auto t0 = std::chrono::high_resolution_clock::now();
// ... CPU work ...
auto t1 = std::chrono::high_resolution_clock::now();
float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
```

### Benchmark Structure

```cpp
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: benchmark <mesh.obj>\n");
        return 1;
    }

    grwm::MeshData mesh;
    grwm::load_mesh(argv[1], mesh);

    const int RUNS = 10;

    // Warm-up run (not timed)
    auto _ = grwm::compute_curvature(mesh);

    // Stage 1: Curvature
    float gpu_curv_ms = 0;
    for (int i = 0; i < RUNS; ++i) {
        // time compute_curvature()
        gpu_curv_ms += elapsed;
    }
    gpu_curv_ms /= RUNS;

    // Stage 2: Feature edges
    // ... similar pattern ...

    // Stage 3: Slots
    // ... similar pattern ...

    // Print results table
    printf("╔══════════════════╦═══════════════╦═══════════════╦══════════╗\n");
    printf("║ Stage            ║ GPU (ms)      ║ CPU (ms)      ║ Speedup  ║\n");
    printf("╠══════════════════╬═══════════════╬═══════════════╬══════════╣\n");
    printf("║ 1. Curvature     ║ %11.2f   ║ %11.2f   ║ %6.1fx   ║\n", ...);
    printf("║ 2. Feature Edges ║ %11.2f   ║ %11.2f   ║ %6.1fx   ║\n", ...);
    printf("║ 3. Slot Sort     ║ %11.2f   ║ %11.2f   ║ %6.1fx   ║\n", ...);
    printf("╚══════════════════╩═══════════════╩═══════════════╩══════════╝\n");
}
```

### Expected Results (from GRWM spec)

| Stage | Crossover Point | Speedup at 1M faces |
|-------|----------------|---------------------|
| Curvature | ~50k vertices | 20-40× |
| Feature Edges | Small meshes | 10-20× |
| Slot Sort | Always | 50-100× |

### CPU Reference Implementations

The benchmark needs simple CPU implementations of each stage for comparison:
- Stage 1 CPU: Sequential cotangent Laplacian with `std::vector` and direct indexing
- Stage 2 CPU: Sequential edge iteration with normal dot product
- Stage 3 CPU: Sequential `std::sort` per face

These can be implemented inline in `benchmark.cpp` or factored into a separate `cpu_reference.cpp` file.

### Memory Reporting

```cpp
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("GPU Memory: %.1f MB used / %.1f MB total\n",
       (total_mem - free_mem) / (1024.0f * 1024.0f),
       total_mem / (1024.0f * 1024.0f));
```

## Acceptance Criteria

- [ ] Loads mesh and runs all three stages
- [ ] Prints timing table with GPU, CPU, and speedup columns
- [ ] Average over 10 runs (with warm-up)
- [ ] Reports peak GPU memory usage
- [ ] Results are reproducible across runs (< 10% variance)
- [ ] Speedup results match expected ranges from spec

## Dependencies

- Feature 10 (curvature pipeline)
- Feature 13 (feature edge pipeline)
- Feature 17 (slot pipeline)
- Feature 02 (mesh loading)
