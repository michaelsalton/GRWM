# Epic 5: Slot Priority & Segmented Sort (Stage 3)

## Purpose

Compute per-face element slot positions and priorities, then sort slots within each face by descending priority. This enables temporally stable LOD transitions — as the LOD level changes, elements appear and disappear at fixed positions rather than all positions reorganizing simultaneously.

## How It Works

Each face has N_max candidate element positions (slots). At runtime, Gravel's task shader activates only the top K slots for the current LOD level. Because slot positions are deterministic (hash-based) and priority-sorted, elements appear and disappear at stable locations.

## Pipeline

```
Face indices + Curvature buffer
        │
        ├──→ ComputeSlotPriorities kernel ──→ SlotEntry[F × N_max] (unsorted)
        │         │
        │         ├── slot_position(face_id, slot_index)  → (u, v)
        │         ├── centerScore = 1 - dist_to_center
        │         ├── curvScore  = barycentric_interp(curvature)
        │         └── jitter     = fract(sin(...))
        │
        └──→ CUB DeviceSegmentedSort ──→ SlotEntry[F × N_max] (sorted by priority desc)
```

## Features

| # | Feature | Focus |
|---|---------|-------|
| 14 | [Slot position hashing](feature-14-slot-position-hashing.md) | Murmur hash, deterministic UV generation |
| 15 | [Priority score computation kernel](feature-15-priority-score-kernel.md) | ComputeSlotPriorities kernel |
| 16 | [CUB segmented sort integration](feature-16-cub-segmented-sort.md) | DeviceSegmentedSort wrapper |
| 17 | [Slots host orchestration](feature-17-slots-host-pipeline.md) | compute_slots() end-to-end |

## Key Parameters

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| slots_per_face (N_max) | 64 | `--slots` |
| w_center | 0.5 | `--w-center` |
| w_curv | 0.4 | `--w-curv` |
| w_jitter | 0.1 | `--w-jitter` |

## Memory Considerations

For N_max=64 and F=200k faces:
- Slot buffer: 200,000 × 64 × 16 bytes = **~195 MB**
- For 1M faces: **~1 GB** (spec recommends `--slots 32` for GPUs with <4GB VRAM)

## Success Criteria

- [ ] Slot positions are deterministic (same face_id + slot_index always produces same UV)
- [ ] Priorities within each face are in non-increasing order after sort
- [ ] Slot positions are in [0, 1] range
- [ ] Higher-priority slots tend to be near face center (centerScore dominates)
- [ ] Expected speedup: 50-100× over CPU std::sort at 200k faces

## Dependencies

- Epic 1 (mesh data and output I/O)
- Epic 3 (curvature buffer from Stage 1 feeds into priority scoring)

## Downstream Dependents

- Epic 6 (slot validation)
