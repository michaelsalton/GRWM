#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "preprocess.h"

#include <cstdio>

namespace grwm {

bool load_mesh(const std::string& obj_path, MeshData& out_mesh) {
    // TODO: parse OBJ with tinyobjloader, populate positions and indices
    fprintf(stderr, "load_mesh: not yet implemented\n");
    return false;
}

} // namespace grwm
