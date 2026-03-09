#include "preprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace grwm {

// CPU reimplementations of hash functions for validation
static uint32_t cpu_murmur_hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 13;
    key *= 0xc2b2ae35u;
    key ^= key >> 16;
    return key;
}

static void cpu_slot_position(uint32_t face_id, uint32_t slot_index,
                              float& u, float& v) {
    uint32_t h = cpu_murmur_hash(face_id ^ (slot_index * 2654435761u));
    u = float(h & 0xFFFF) / 65535.0f;
    v = float(h >> 16) / 65535.0f;
}

// --- Feature 18: Sphere curvature validation ---

bool validate_curvature(const MeshData& mesh,
                        const std::vector<float>& curvature)
{
    if (mesh.vertex_count == 0 || curvature.size() != mesh.vertex_count) {
        fprintf(stderr, "validate_curvature: size mismatch\n");
        return false;
    }

    // Compute centroid
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (uint32_t i = 0; i < mesh.vertex_count; ++i) {
        cx += mesh.positions[i * 3 + 0];
        cy += mesh.positions[i * 3 + 1];
        cz += mesh.positions[i * 3 + 2];
    }
    cx /= mesh.vertex_count;
    cy /= mesh.vertex_count;
    cz /= mesh.vertex_count;

    // Compute average distance to centroid (radius estimate)
    float sum_dist = 0.0f;
    float min_dist = 1e30f, max_dist = 0.0f;
    for (uint32_t i = 0; i < mesh.vertex_count; ++i) {
        float dx = mesh.positions[i * 3 + 0] - cx;
        float dy = mesh.positions[i * 3 + 1] - cy;
        float dz = mesh.positions[i * 3 + 2] - cz;
        float d = sqrtf(dx * dx + dy * dy + dz * dz);
        sum_dist += d;
        min_dist = std::min(min_dist, d);
        max_dist = std::max(max_dist, d);
    }
    float R = sum_dist / mesh.vertex_count;

    // Check if mesh is approximately spherical (distance variance < 5%)
    float radius_variation = (max_dist - min_dist) / (R + 1e-10f);
    if (radius_variation > 0.1f) {
        printf("  Curvature validation: mesh is not spherical (radius variation %.1f%%), skipping\n",
               radius_variation * 100.0f);
        return true;
    }

    float expected = 1.0f / R;

    // Compute statistics
    float mae = 0.0f;
    float min_curv = curvature[0], max_curv = curvature[0], sum_curv = 0.0f;
    for (uint32_t i = 0; i < mesh.vertex_count; ++i) {
        float c = curvature[i];
        mae += fabsf(c - expected);
        min_curv = std::min(min_curv, c);
        max_curv = std::max(max_curv, c);
        sum_curv += c;
    }
    mae /= mesh.vertex_count;
    float mean_curv = sum_curv / mesh.vertex_count;

    float relative_error = mae / (expected + 1e-10f);
    bool pass = relative_error < 0.05f;

    printf("  Curvature validation: R=%.4f, expected H=%.4f\n", R, expected);
    printf("    Mean=%.4f, Min=%.4f, Max=%.4f, MAE=%.6f (%.2f%%)\n",
           mean_curv, min_curv, max_curv, mae, relative_error * 100.0f);
    printf("    Result: %s\n", pass ? "PASS" : "FAIL");

    return pass;
}

// --- Features 19 + 20: Slot sort order validation ---

bool validate_slots(const std::vector<SlotEntry>& slots,
                    uint32_t face_count, uint32_t slots_per_face)
{
    uint32_t total = face_count * slots_per_face;
    if (slots.size() != total) {
        fprintf(stderr, "validate_slots: expected %u entries, got %zu\n",
                total, slots.size());
        return false;
    }

    // Sample faces deterministically
    uint32_t samples = std::min(face_count, 100u);
    uint32_t step = (samples > 0) ? face_count / samples : 1;

    uint32_t sort_violations = 0;
    uint32_t hash_violations = 0;

    for (uint32_t s = 0; s < samples; ++s) {
        uint32_t fid = s * step;
        uint32_t base = fid * slots_per_face;

        // Check sort order (descending priority)
        for (uint32_t i = 1; i < slots_per_face; ++i) {
            if (slots[base + i].priority > slots[base + i - 1].priority) {
                sort_violations++;
                if (sort_violations <= 5) {
                    printf("    Sort violation: face %u, slot %u: %.4f > %.4f\n",
                           fid, i, slots[base + i].priority,
                           slots[base + i - 1].priority);
                }
            }
        }

        // Check that slot positions match deterministic hash
        for (uint32_t i = 0; i < slots_per_face; ++i) {
            uint32_t sid = slots[base + i].slot_index;
            float expected_u, expected_v;
            cpu_slot_position(fid, sid, expected_u, expected_v);

            float du = fabsf(slots[base + i].u - expected_u);
            float dv = fabsf(slots[base + i].v - expected_v);
            if (du > 1e-5f || dv > 1e-5f) {
                hash_violations++;
                if (hash_violations <= 5) {
                    printf("    Hash violation: face %u, slot %u: "
                           "got (%.6f,%.6f), expected (%.6f,%.6f)\n",
                           fid, sid, slots[base + i].u, slots[base + i].v,
                           expected_u, expected_v);
                }
            }
        }
    }

    printf("  Slot validation: sampled %u faces\n", samples);
    printf("    Sort violations: %u, Hash violations: %u\n",
           sort_violations, hash_violations);

    bool pass = (sort_violations == 0 && hash_violations == 0);
    printf("    Result: %s\n", pass ? "PASS" : "FAIL");

    return pass;
}

} // namespace grwm
