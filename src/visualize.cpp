#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace grwm {

// --- Binary output helpers ---

static bool write_bin(const std::string& path,
                      const PreprocessHeader& header,
                      const void* data,
                      size_t data_bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(PreprocessHeader));
    if (data_bytes > 0) {
        out.write(reinterpret_cast<const char*>(data), data_bytes);
    }
    if (!out.good()) {
        fprintf(stderr, "Write error on %s\n", path.c_str());
        return false;
    }
    printf("  Wrote %s (%zu bytes)\n", path.c_str(),
           sizeof(PreprocessHeader) + data_bytes);
    return true;
}

bool write_curvature_bin(const std::string& path,
                         const PreprocessHeader& header,
                         const std::vector<float>& curvature)
{
    return write_bin(path, header, curvature.data(),
                     curvature.size() * sizeof(float));
}

bool write_features_bin(const std::string& path,
                        const PreprocessHeader& header,
                        const std::vector<uint8_t>& flags)
{
    return write_bin(path, header, flags.data(),
                     flags.size() * sizeof(uint8_t));
}

bool write_slots_bin(const std::string& path,
                     const PreprocessHeader& header,
                     const std::vector<SlotEntry>& slots)
{
    return write_bin(path, header, slots.data(),
                     slots.size() * sizeof(SlotEntry));
}

// --- Visualization (Epic 6, stubs for now) ---

bool write_curvature_ply(const std::string& path,
                         const MeshData& mesh,
                         const std::vector<float>& curvature)
{
    // TODO: implement in Epic 6 (Feature 21)
    fprintf(stderr, "write_curvature_ply: not yet implemented\n");
    return false;
}

bool write_feature_edges_obj(const std::string& path,
                             const MeshData& mesh,
                             const std::vector<uint8_t>& edge_flags)
{
    // TODO: implement in Epic 6 (Feature 21)
    fprintf(stderr, "write_feature_edges_obj: not yet implemented\n");
    return false;
}

} // namespace grwm
