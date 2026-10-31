"""Stage 3. Control map (+ optional prompt) -> object image, using SD + ControlNet."""
import numpy as np
import torch

from .config import resolve_path
from .device import torch_device
from .errors import SketchToImageError, WeightsError

_pipe = None


def build_prompt(prompt, cfg):
    settings = cfg["sketch_to_image"]
    subject = (prompt or "").strip() or settings["default_subject"]
    return settings["prompt_template"].format(subject=subject)


def weights_path(name, cfg):
    path = resolve_path(cfg["paths"]["weights_dir"]) / name
    if not path.is_dir():
        raise WeightsError(f"missing weights: {path}", hint="run scripts/download_weights.py")
    return path


def _load(cfg, device, dtype):
    global _pipe
    if _pipe is not None:
        return _pipe
    import diffusers

    settings = cfg["sketch_to_image"]
    variant = settings["weights_variant"]
    controlnet = diffusers.ControlNetModel.from_pretrained(
        weights_path(settings["controlnet_weights"], cfg), torch_dtype=dtype, variant=variant
    )
    pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
        weights_path(settings["base_weights"], cfg),
        controlnet=controlnet, torch_dtype=dtype, variant=variant,
        safety_checker=None, requires_safety_checker=False,
    )
    scheduler_cls = getattr(diffusers, settings["scheduler"])
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    if cfg["runtime"]["attention_slicing"]:
        pipe.enable_attention_slicing()
    if cfg["runtime"]["cpu_offload"] and device.startswith("cuda"):
        pipe.enable_model_cpu_offload(gpu_id=int(device.split(":")[1]))
    else:
        pipe.to(torch_device(device))
    _pipe = pipe
    return pipe


def unload():
    global _pipe
    _pipe = None


def infer_subject(clean, cfg, device):
    """Name the sketch with CLIP. Returns (label or None, confidence). Weights are not kept loaded."""
    from transformers import CLIPModel, CLIPProcessor

    settings = cfg["sketch_to_image"]
    path = weights_path(settings["clip_weights"], cfg)
    model = CLIPModel.from_pretrained(path).to(torch_device(device))
    processor = CLIPProcessor.from_pretrained(path)
    labels = settings["subject_labels"]
    texts = [settings["subject_template"].format(label) for label in labels]
    inputs = processor(text=texts, images=clean, return_tensors="pt", padding=True).to(torch_device(device))
    with torch.no_grad():
        scores = model(**inputs).logits_per_image.softmax(dim=-1)[0]
    best = int(scores.argmax())
    confidence = float(scores[best])
    return (labels[best] if confidence >= settings["subject_min_confidence"] else None), confidence


def generate(control, prompt, cfg, device, dtype_name):
    settings = cfg["sketch_to_image"]
    pipe = _load(cfg, device, getattr(torch, dtype_name))
    image = pipe(
        build_prompt(prompt, cfg),
        image=control,
        negative_prompt=settings["negative_prompt"],
        num_inference_steps=settings["steps"],
        guidance_scale=settings["guidance_scale"],
        controlnet_conditioning_scale=settings["controlnet_scale"],
        generator=torch.Generator("cpu").manual_seed(cfg["runtime"]["seed"]),
    ).images[0]
    if np.asarray(image).std() < settings["min_image_std"]:
        raise SketchToImageError(
            "the generated image is blank",
            hint="this often means a numeric overflow; try runtime.dtype=float32",
        )
    return image
