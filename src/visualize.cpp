#include "preprocess.h"

#include <cstdio>
#include <fstream>

namespace grwm {

bool write_curvature_bin(const std::string& path,
                         const PreprocessHeader& header,
                         const std::vector<float>& curvature)
{
    // TODO: write header + float array to binary file
    fprintf(stderr, "write_curvature_bin: not yet implemented\n");
    return false;
}

bool write_features_bin(const std::string& path,
                        const PreprocessHeader& header,
                        const std::vector<uint8_t>& flags)
{
    // TODO: write header + uint8 array to binary file
    fprintf(stderr, "write_features_bin: not yet implemented\n");
    return false;
}

bool write_slots_bin(const std::string& path,
                     const PreprocessHeader& header,
                     const std::vector<SlotEntry>& slots)
{
    // TODO: write header + SlotEntry array to binary file
    fprintf(stderr, "write_slots_bin: not yet implemented\n");
    return false;
}

bool write_curvature_ply(const std::string& path,
                         const MeshData& mesh,
                         const std::vector<float>& curvature)
{
    // TODO: write PLY with per-vertex color mapped from curvature
    // (blue = low, red = high)
    fprintf(stderr, "write_curvature_ply: not yet implemented\n");
    return false;
}

bool write_feature_edges_obj(const std::string& path,
                             const MeshData& mesh,
                             const std::vector<uint8_t>& edge_flags)
{
    // TODO: write OBJ containing only feature edge line segments
    fprintf(stderr, "write_feature_edges_obj: not yet implemented\n");
    return false;
}

} // namespace grwm
