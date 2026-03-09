#include "preprocess.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace grwm {

bool validate_curvature(const MeshData& mesh,
                        const std::vector<float>& curvature)
{
    // TODO: validate against analytical ground truth
    // - Sphere of radius R: mean curvature should be 1/R
    // - Cylinder of radius R: curved surface 1/(2R), flat caps ~0
    fprintf(stderr, "validate_curvature: not yet implemented\n");
    return true;
}

bool validate_slots(const std::vector<SlotEntry>& slots,
                    uint32_t face_count, uint32_t slots_per_face)
{
    // TODO: sample random faces and verify:
    // 1. Priority values are in non-increasing order within each face
    // 2. Slot positions match deterministic hash function output
    fprintf(stderr, "validate_slots: not yet implemented\n");
    return true;
}

} // namespace grwm
