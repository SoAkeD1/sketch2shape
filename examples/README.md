# Sample results

Real runs from `config.yaml` defaults on an AMD Radeon RX 9070 XT. Nothing was hand-edited.

| Sample | Prompt | CLIP read | Sketch match | Silhouette IoU | Faces | Time | .glb | Seed |
|---|---|---|---|---|---|---|---|---|
| [flower](flower/) | none | a flower | 87% | 0.73 | 42,044 | 44 s | 2.4 MB | 43 |
| [vase](vase/) | none | a bottle | 88% | 0.79 | 100,000 | 34 s | 4.6 MB | 42 |
| [vase_blue_prompt](vase_blue_prompt/) | a blue glass vase | n/a | 87% | 0.81 | 100,000 | 30 s | 5.0 MB | 42 |
| [table_best_of_3](table_best_of_3/) | none | a table | 91% | 0.58 | 84,156 | 35 s | 3.2 MB | 142 |

**Selection:** every sample shown is the output of its first run here, not picked from repeated runs. The exception is `table_best_of_3`, which is the top-ranked result of the **Generate 3 variants** feature.

**Sketch source:** `flower` and `vase` were hand-drawn by the user in the app. `table_best_of_3` was drawn with mouse strokes by the AI assistant while testing the app. These sketches were chosen because they already existed, so they are not a random test set.

**Input files:** `sketch.png` is the cleaned sketch, because the app did not store the raw canvas drawing. These runs took those cleaned sketches as input.

## Files in each folder

| File | What it is |
|---|---|
| `sketch.png` | Cleaned input sketch |
| `generated.png` | Image from Stable Diffusion + ControlNet |
| `sketch_match.png` | Strokes over the model: green on the model, red missed |
| `views.png` | Model rendered from 8 sides |
| `model.glb` | 3D model with texture |
| `run_summary.json` | Metrics, attempts, timings and validation |
| `obj/` | `.obj`, `.mtl` and texture (`flower` only) |

Open a `model.glb` at https://gltf-viewer.donmccurdy.com/ or in Blender.

