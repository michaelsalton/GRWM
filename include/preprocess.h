#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace grwm {

// --- Binary format constants ---
constexpr uint32_t GRWM_MAGIC               = 0x47525650; // "GRVP"
constexpr uint32_t GRWM_VERSION             = 1;
constexpr uint32_t DEFAULT_SLOTS_PER_FACE   = 64;
constexpr float    DEFAULT_FEATURE_THRESHOLD_DEG = 30.0f;

// --- Binary file header (32 bytes, matches Gravel loader) ---
struct PreprocessHeader {
    uint32_t magic;          // GRWM_MAGIC
    uint32_t version;        // GRWM_VERSION
    uint32_t vertex_count;
    uint32_t face_count;
    uint32_t edge_count;
    uint32_t slots_per_face;
    uint32_t padding[2];
};
static_assert(sizeof(PreprocessHeader) == 32, "Header must be 32 bytes");

// --- Slot entry (16 bytes, matches Gravel SSBO layout) ---
struct SlotEntry {
    float    u;
    float    v;
    float    priority;
    uint32_t slot_index;
};
static_assert(sizeof(SlotEntry) == 16, "SlotEntry must be 16 bytes");

// --- Mesh data (CPU-side, loaded from OBJ) ---
struct MeshData {
    std::vector<float>    positions;   // flat: x,y,z,x,y,z,...
    std::vector<uint32_t> indices;     // flat: i0,i1,i2,i0,i1,i2,...
    uint32_t vertex_count = 0;
    uint32_t face_count   = 0;
};

// --- Pipeline configuration ---
struct PipelineConfig {
    std::string input_path;
    std::string output_dir    = "./output/";
    uint32_t    slots_per_face       = DEFAULT_SLOTS_PER_FACE;
    float       feature_threshold_deg = DEFAULT_FEATURE_THRESHOLD_DEG;
    bool        validate             = false;
    bool        vis_curvature        = false;
    float       w_center             = 0.5f;
    float       w_curv               = 0.4f;
    float       w_jitter             = 0.1f;
};

// --- Mesh loading ---
bool load_mesh(const std::string& obj_path, MeshData& out_mesh);

// --- Stage 1: Per-vertex mean curvature ---
std::vector<float> compute_curvature(const MeshData& mesh);

// --- Stage 2: Feature edge detection ---
std::vector<uint8_t> detect_feature_edges(
    const MeshData& mesh,
    float threshold_degrees);

// --- Stage 3: Slot priority computation and sort ---
std::vector<SlotEntry> compute_slots(
    const MeshData& mesh,
    const std::vector<float>& curvature,
    uint32_t slots_per_face,
    float w_center, float w_curv, float w_jitter);

// --- Output writing ---
bool write_curvature_bin(const std::string& path,
                         const PreprocessHeader& header,
                         const std::vector<float>& curvature);
bool write_features_bin(const std::string& path,
                        const PreprocessHeader& header,
                        const std::vector<uint8_t>& flags);
bool write_slots_bin(const std::string& path,
                     const PreprocessHeader& header,
                     const std::vector<SlotEntry>& slots);

// --- Validation ---
bool validate_curvature(const MeshData& mesh,
                        const std::vector<float>& curvature);
bool validate_slots(const std::vector<SlotEntry>& slots,
                    uint32_t face_count, uint32_t slots_per_face);

// --- Visualization ---
bool write_curvature_ply(const std::string& path,
                         const MeshData& mesh,
                         const std::vector<float>& curvature);
bool write_feature_edges_obj(const std::string& path,
                             const MeshData& mesh,
                             const std::vector<uint8_t>& edge_flags);

} // namespace grwm
