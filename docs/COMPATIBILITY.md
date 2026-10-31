# Model × backend compatibility

Tested 2026-10-31 with one in-memory test line drawing, not a user sketch. Quality on real sketches is untested until Phase 1.

**Local test machine:** Windows 11 25H2, RX 9070 XT 16 GB, Python 3.11.9, torch 2.14.0+rocm10.1.0, driver 25.10.36.

## Results
| Model | Stage | Local ROCm (Windows) | DirectML | CUDA (Colab/Kaggle) | CPU | License |
|---|---|---|---|---|---|---|
| **SD 1.5 + ControlNet Scribble** | Sketch→image | ✅ Works: 25 steps 5.4 s warm, 3.13 GB peak | Not tested | Not tested | Not tested | OpenRAIL-M |
| SDXL + xinsir scribble | Sketch→image | Not tested | Not tested | Not tested | Not tested | Apache-2.0 (ControlNet) |
| **rembg isnet-general-use** | Background | ✅ Works on CPU ONNX: 2.4 s | n/a | Not tested | ✅ Same run | See ATTRIBUTION |
| **TripoSR** | Image→3D | ✅ Works with 2 adapters: 14.1 s, 2.35 GB peak, 192k faces | Not tested | Not tested | Not tested | MIT |
| Stable Fast 3D | Image→3D | Not tested; needs compiled kernels and a C++ compiler | Not tested | Not tested | Documented | Community (under $1M revenue); gated |
| Hunyuan3D-2 mini | Image→3D | Not tested; texture needs compiled extensions | Not tested | Not tested | Not tested | Not checked |
| InstantMesh | Image→3D | ❌ Not attempted: documents CUDA 12.1 setup, Linux commands | ❌ Not attempted | Not tested | Not attempted | Apache-2.0 |
| Unique3D | Image→3D | ❌ Not attempted: documented for CUDA 12.1 | ❌ Not attempted | Not tested | Not attempted | MIT |
| **CLIP ViT-B/32** | Metrics | ✅ Loads and runs: 1.0 s, 0.60 GB | Not tested | Not tested | Not tested | MIT |

✅ means run on this machine. "Not attempted" is a judgment from each project's install docs, not a failed test.

## Choices
| Stage | Primary | Fallback |
|---|---|---|
| Sketch→image | SD 1.5 + Scribble | Change ControlNet type (lineart/canny), Phase 1 if needed |
| Background | isnet-general-use | u2netp (smaller, usually rougher) |
| Image→3D | TripoSR | Not chosen yet. Stable Fast 3D or Hunyuan3D-2 mini need testing, possibly on `remote_gpu` |
| Backend | `local` (ROCm) | `remote_gpu` (notebook written, not yet run) |

## Things needed for local ROCm
1. **GPU index:** index 0 on this PC is the Ryzen iGPU. It reports as available, then fails with `invalid kernel file`. `device.py` picks the usable GPU with the most memory.
2. **torchmcubes:** needs a compiler, none installed. The probe replaced it with scikit-image marching cubes. Output mesh is correctly oriented and watertight.
3. **transformers 5:** renamed the ViT layers, so TripoSR's checkpoint will not load unmodified. Renaming the keys at load time gives `All keys matched`. Pinning transformers 4 breaks diffusers, so don't.

Adapters 2 and 3 live in `image_to_3d.py`. It also runs TripoSR's texture color lookup on the GPU, because the upstream version only works on CPU.

## Known risks
- AMD's Windows ROCm is new. AMD lists Adrenalin 26.8.1 for ROCm 10.0; this PC ran 25.10.36 without a failure so far. Smart App Control is on, and it also caused no failure so far.
- TripoSR reads a config from Hugging Face at startup. `download_weights.py` now caches it, but offline behavior is still untested.
- The test mug came out solid, not hollow. Expect guessed hidden geometry.
- Retries are not built yet (Phase 4).
