# Attribution and Disclosure

Every model, library, service and AI assistant this project uses.

## AI assistant log
| When | Assistant | Used for | Why |
|---|---|---|---|
| 2026-10-31, Phase 0 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Repo skeleton, config loader, device report, weight downloader, launcher notebook, model and backend compatibility research | Fast scaffolding and checking license and hardware constraints within the 24-hour limit |
| 2026-10-31, Phase 1 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Pipeline stage modules, TripoSR compatibility adapters, debugging, test runs | Fast implementation and diagnosis of library and hardware conflicts |
| 2026-10-31, Phase 2 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Mesh cleanup, export, validation, run metadata, test runs | Fast implementation and checking results in an independent viewer |
| 2026-10-31, Phase 3 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Gradio app, CLI, testing the app in a browser | Fast implementation and end-to-end checks of the user flow |
| 2026-11-01, Phase 4 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Metrics, retry logic, test batches, threshold selection | Fast measurement and diagnosis across many test runs |
| 2026-11-01, Phase 5 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | README, clean-install test | Checking setup steps by following them literally |
| 2026-11-01, Phase 6 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Project report, slides outline, repository checks | Drafting from measured results within the time limit |
| 2026-11-01, Phase 7 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Sketch match view, ranked variants, testing them | Proposed the ideas; the user chose which to build |
| 2026-11-01, Phase 9 | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Generated and checked the sample results in `examples/` | Packaging real runs for the required output artifact |
| 2026-11-01, Phase 8 | Claude Code (Claude Opus 5.5, Anthropic), desktop app, with Claude Design | Claude Design produced the Living Notebook design from the team's brief. Claude Code implemented it, and wrote the regression test | A polished, engaging interface within the time limit |
| 2026-11-01, demo video | Claude Code (Claude Opus 5.5, Anthropic), desktop app | Recorded the demo by controlling the real app, wrote the narration and edited the video | Producing a real recording within the time limit |

Demo video tools: Kokoro-82M text-to-speech (https://huggingface.co/hexgrad/Kokoro-82M, Apache-2.0) for narration, Playwright and Chrome for recording, FFmpeg for editing. None of these are part of the app.

The team directed the work, chose priorities, drew the user test sketches, and made the final edit of the demo video.

## Responsible use notes
- Stable Diffusion's optional safety checker is turned off in `sketch_to_image.py`. It often misfires on plain object renders, and its extra model adds memory use. Outputs are not filtered, so users stay responsible for what they generate, under the OpenRAIL-M use restrictions.

## Pretrained models
| Model | Link | License | Training data (as stated by publisher) | Status |
|---|---|---|---|---|
| Stable Diffusion v1.5 | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 | CreativeML OpenRAIL-M | LAION-2B (en) and subsets | Used |
| ControlNet v1.1 Scribble | https://huggingface.co/lllyasviel/control_v11p_sd15_scribble | CreativeML OpenRAIL-M | Not named; scribbles synthesized from images | Used |
| TripoSR | https://huggingface.co/stabilityai/TripoSR | MIT | Curated Objaverse subset, rendered | Used |
| IS-Net general use, via rembg | https://github.com/danielgatis/rembg, https://github.com/xuebinqin/DIS | IS-Net code Apache-2.0; license of these exact weights not separately stated | IS-Net academic model: DIS5K. Training data of these exact weights not stated | Used |
| CLIP ViT-B/32 | https://huggingface.co/openai/clip-vit-base-patch32 | MIT | See model card | Used: names the sketch when there is no prompt, and scores results |

## Libraries
Exact versions, including indirect dependencies, are pinned in `requirements.txt` and `requirements-rocm.txt`.

Used directly: PyTorch (AMD ROCm build or CUDA build), diffusers, transformers, accelerate, huggingface_hub, safetensors, rembg, onnxruntime, OpenCV, scikit-image, NumPy, Pillow, trimesh, fast-simplification, xatlas, moderngl, omegaconf, einops, Gradio, PyYAML, psutil.

TripoSR source code (https://github.com/VAST-AI-Research/TripoSR, MIT, commit in `config.yaml`) is downloaded at setup and is not copied into this repository. `image_to_3d.py` adapts how it loads, without changing its files.

The test viewer used once to check `.glb` files, Google's `<model-viewer>`, is not part of the project.

Fonts: Inter and JetBrains Mono, loaded from Google Fonts (`fonts.googleapis.com`), under the SIL Open Font License.

## External services
- Hugging Face Hub: weight downloads, plus a config check when TripoSR loads.
- GitHub: TripoSR source and the rembg model file.
- Google Colab or Kaggle: optional `remote_gpu` backend.
- AMD wheel index (`stable.repo.amd.com`): PyTorch for local AMD GPUs.

## Datasets
No custom dataset, and no training or fine-tuning. Pretrained models only. Their training data is listed above and in `REPORT.md`.

## License
This project's own code is under the MIT License (see `LICENSE`). Pretrained models and third-party code keep their own licenses, listed above.

## Pre-existing code
None. All project code was written during the hackathon.
