"""Stage 4. Object image -> RGBA with the object isolated, centered and padded."""
import cv2
import numpy as np
from PIL import Image

from .errors import BackgroundError

_session = None
_session_model = None


def _get_session(model):
    global _session, _session_model
    if _session is None or _session_model != model:
        from rembg import new_session
        _session = new_session(model)
        _session_model = model
    return _session


def unload():
    global _session, _session_model
    _session = None
    _session_model = None


def _drop_small_components(mask, min_ratio):
    """Remove stray bits, but keep real parts such as a thin lamp stem cut off from its shade."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, 1 + np.flatnonzero(areas >= min_ratio * areas.max()))


def isolate(image, cfg):
    """Returns (centered RGBA, foreground mask in the frame of the input image)."""
    from rembg import remove

    settings = cfg["background"]
    rgba = np.array(remove(image.convert("RGB"), session=_get_session(settings["rembg_model"])))
    mask = rgba[..., 3] > settings["alpha_level"]
    mask = _drop_small_components(mask, settings["min_component_ratio"])
    rgba[..., 3] = np.where(mask, rgba[..., 3], 0)
    coverage = mask.mean()
    if coverage < settings["min_coverage"]:
        raise BackgroundError(
            f"only {coverage:.1%} of the image was kept as the object",
            hint="try another seed or a higher sketch_to_image.controlnet_scale",
        )
    ys, xs = np.nonzero(mask)
    crop = Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    size = settings["output_size"]
    scale = settings["foreground_ratio"] * size / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return result, mask
