#include "preprocess.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static void print_usage(const char* program) {
    printf("Usage: %s <input.obj> [options]\n\n", program);
    printf("Options:\n");
    printf("  --output <dir>              Output directory (default: ./output/)\n");
    printf("  --slots <N>                 Slots per face (default: 64)\n");
    printf("  --feature-threshold <deg>   Feature edge threshold in degrees (default: 30.0)\n");
    printf("  --validate                  Run validation checks\n");
    printf("  --vis-curvature             Write curvature visualization PLY\n");
    printf("  --w-center <f>              Center priority weight (default: 0.5)\n");
    printf("  --w-curv <f>                Curvature priority weight (default: 0.4)\n");
    printf("  --w-jitter <f>              Jitter priority weight (default: 0.1)\n");
}

static bool parse_args(int argc, char* argv[], grwm::PipelineConfig& config) {
    if (argc < 2) return false;

    config.input_path = argv[1];
    if (config.input_path == "--help" || config.input_path == "-h") return false;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            config.slots_per_face = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--feature-threshold") == 0 && i + 1 < argc) {
            config.feature_threshold_deg = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--validate") == 0) {
            config.validate = true;
        } else if (strcmp(argv[i], "--vis-curvature") == 0) {
            config.vis_curvature = true;
        } else if (strcmp(argv[i], "--w-center") == 0 && i + 1 < argc) {
            config.w_center = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--w-curv") == 0 && i + 1 < argc) {
            config.w_curv = static_cast<float>(atof(argv[++i]));
        } else if (strcmp(argv[i], "--w-jitter") == 0 && i + 1 < argc) {
            config.w_jitter = static_cast<float>(atof(argv[++i]));
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    grwm::PipelineConfig config;

    if (!parse_args(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }

    printf("GRWM - CUDA Mesh Preprocessing Pipeline\n");
    printf("Input:  %s\n", config.input_path.c_str());
    printf("Output: %s\n", config.output_dir.c_str());
    printf("Slots:  %u\n", config.slots_per_face);

    // Load mesh
    grwm::MeshData mesh;
    if (!grwm::load_mesh(config.input_path, mesh)) {
        fprintf(stderr, "Failed to load mesh: %s\n", config.input_path.c_str());
        return 1;
    }
    printf("Loaded mesh: %u vertices, %u faces\n", mesh.vertex_count, mesh.face_count);

    // Create output directory
    fs::create_directories(config.output_dir);

    // Stage 1: Curvature
    printf("Stage 1: Computing curvature...\n");
    auto curvature = grwm::compute_curvature(mesh);

    // Stage 2: Feature edges
    printf("Stage 2: Detecting feature edges...\n");
    auto features = grwm::detect_feature_edges(mesh, config.feature_threshold_deg);

    // Stage 3: Slot priorities
    printf("Stage 3: Computing slot priorities...\n");
    auto slots = grwm::compute_slots(mesh, curvature, config.slots_per_face,
                                     config.w_center, config.w_curv, config.w_jitter);

    // Write output
    grwm::PreprocessHeader header{};
    header.magic         = grwm::GRWM_MAGIC;
    header.version       = grwm::GRWM_VERSION;
    header.vertex_count  = mesh.vertex_count;
    header.face_count    = mesh.face_count;
    header.edge_count    = 0; // TODO: compute from half-edge mesh
    header.slots_per_face = config.slots_per_face;

    grwm::write_curvature_bin(config.output_dir + "/curvature.bin", header, curvature);
    grwm::write_features_bin(config.output_dir + "/features.bin", header, features);
    grwm::write_slots_bin(config.output_dir + "/slots.bin", header, slots);

    // Optional validation
    if (config.validate) {
        printf("Running validation...\n");
        grwm::validate_curvature(mesh, curvature);
        grwm::validate_slots(slots, mesh.face_count, config.slots_per_face);
    }

    // Optional visualization
    if (config.vis_curvature) {
        grwm::write_curvature_ply(config.output_dir + "/curvature_vis.ply", mesh, curvature);
    }

    printf("Done.\n");
    return 0;
}
