# GRWM: CUDA Mesh Preprocessing Pipeline

**Geometry Resampling and Weighting for Meshes**

A parallel GPU preprocessing tool for procedural geometry systems. Companion project to [Gravel](https://github.com/michaelsalton/Gravel).

GRWM accepts a raw triangle mesh (OBJ) and produces three binary buffer files consumed by Gravel at load time:

1. **Per-vertex mean curvature** — cotangent Laplacian via cuSPARSE
2. **Per-face feature edge flags** — dihedral angle detection
3. **Per-face priority-sorted slot grids** — segmented sort via CUB

## Prerequisites

- CUDA Toolkit 12+
- CMake 3.20+
- C++17 compiler (MSVC 2019+, GCC 9+, or Clang 10+)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

## Build

```bash
cmake -B build
cmake --build build
```

## Usage

```bash
# Preprocess a mesh (run once)
./build/cuda_preprocess assets/bunny.obj --output assets/bunny_preprocess/ --slots 64

# Run Gravel with preprocessed buffers
./Gravel --mesh assets/bunny.obj --preprocess assets/bunny_preprocess/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output <dir>` | `./output/` | Output directory for binary files |
| `--slots <N>` | `64` | Number of element slots per face |
| `--feature-threshold <deg>` | `30.0` | Dihedral angle threshold for feature edges |
| `--validate` | off | Run validation against analytical ground truth |
| `--vis-curvature` | off | Write curvature visualization PLY file |

## Documentation

See [docs/GRWM.pdf](docs/GRWM.pdf) for the full technical specification.
