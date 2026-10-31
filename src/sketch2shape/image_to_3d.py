"""Stage 5. Centered RGBA -> mesh. Backends share one interface; config picks primary and fallback."""
import logging
import re
import sys
import types
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from . import mesh_post
from .config import resolve_path
from .device import free_gpu_memory, torch_device
from .errors import ImageTo3DError, WeightsError

log = logging.getLogger(__name__)


@dataclass
class Reconstruction:
    mesh: trimesh.Trimesh
    model: str
    textured: bool
    notes: list = field(default_factory=list)
    clean_stats: dict = field(default_factory=dict)


def _install_marching_cubes_adapter():
    """torchmcubes needs a C++ build. Without it, serve the same call from scikit-image."""
    try:
        import torchmcubes  # noqa: F401
        return
    except ImportError:
        pass
    from skimage import measure

    def marching_cubes(volume, threshold):
        verts, faces, _, _ = measure.marching_cubes(volume.detach().cpu().numpy(), threshold)
        # torchmcubes returns z,y,x vertices with the opposite face winding
        return (
            torch.from_numpy(verts[:, ::-1].copy()).float(),
            torch.from_numpy(faces[:, ::-1].copy()).long(),
        )

    module = types.ModuleType("torchmcubes")
    module.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = module


# transformers 5 renamed the ViT layers inside TripoSR's image encoder
_VIT_KEY_RENAMES = [
    (r"model\.encoder\.layer\.", "model.layers."),
    (r"attention\.attention\.query", "attention.q_proj"),
    (r"attention\.attention\.key", "attention.k_proj"),
    (r"attention\.attention\.value", "attention.v_proj"),
    (r"attention\.output\.dense", "attention.o_proj"),
    (r"intermediate\.dense", "mlp.fc1"),
    (r"(layers\.\d+\.)output\.dense", r"\1mlp.fc2"),
]


def _rename_vit_key(key):
    if key.startswith("image_tokenizer."):
        for pattern, replacement in _VIT_KEY_RENAMES:
            key = re.sub(pattern, replacement, key)
    return key


class TripoSRBackend:
    name = "triposr"

    def __init__(self, cfg, device):
        self.settings = cfg["image_to_3d"]["triposr"]
        self.device = torch_device(device)
        source = resolve_path(cfg["paths"]["third_party_dir"]) / self.name
        weights = resolve_path(cfg["paths"]["weights_dir"]) / self.settings["weights"]
        if not source.is_dir():
            raise WeightsError(f"missing TripoSR source: {source}", hint="run scripts/download_weights.py")
        if not weights.is_dir():
            raise WeightsError(f"missing weights: {weights}", hint="run scripts/download_weights.py")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        _install_marching_cubes_adapter()

        from omegaconf import OmegaConf
        from tsr.system import TSR

        model_cfg = OmegaConf.load(weights / self.settings["config_name"])
        OmegaConf.resolve(model_cfg)
        self.model = TSR(model_cfg)
        state = torch.load(weights / self.settings["weight_name"], map_location="cpu", weights_only=True)
        if any("model.layers." in key for key in self.model.state_dict()):
            state = {_rename_vit_key(key): value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.renderer.set_chunk_size(self.settings["chunk_size"])
        self.model.to(self.device)

    def _composite(self, rgba):
        array = np.asarray(rgba).astype(np.float32) / 255
        rgb, alpha = array[..., :3], array[..., 3:]
        flat = rgb * alpha + (1 - alpha) * self.settings["background_gray"]
        return Image.fromarray((flat * 255).astype(np.uint8))

    def _bleed(self, image, empty):
        """Grow island colors outward, one pixel per pass, into the unused atlas area."""
        kernel = np.ones((3, 3), np.uint8)
        filled = (~empty).astype(np.uint8)
        for _ in range(self.settings["texture_bleed_px"]):
            weight = cv2.dilate(filled, kernel)
            grown = cv2.dilate(image * filled[..., None], kernel)
            new = (weight == 1) & (filled == 0)
            image[new] = grown[new]
            filled = weight
        return image

    def _bake(self, mesh, scene_code):
        from tsr.bake_texture import make_atlas, rasterize_position_atlas

        resolution = self.settings["texture_resolution"]
        padding = self.settings["texture_padding"]
        atlas = make_atlas(mesh, resolution, padding)
        positions = rasterize_position_atlas(
            mesh, atlas["vmapping"], atlas["indices"], atlas["uvs"], resolution, padding
        ).reshape(-1, 4)
        # TripoSR's own color lookup only runs on CPU, so query the GPU model directly
        query = torch.from_numpy(positions[:, :3].copy()).to(scene_code.device)
        with torch.no_grad():
            colors = self.model.renderer.query_triplane(self.model.decoder, query, scene_code)["color"]
        colors = colors.cpu().numpy()
        empty = positions[:, 3] == 0
        colors[empty] = 0
        image = (colors.reshape(resolution, resolution, 3) * 255).astype(np.uint8)
        image = self._bleed(image, empty.reshape(resolution, resolution))
        # The atlas has its origin at the bottom; images start at the top
        texture = Image.fromarray(image).transpose(Image.FLIP_TOP_BOTTOM)
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0)
        visual = trimesh.visual.TextureVisuals(uv=atlas["uvs"], material=material)
        return trimesh.Trimesh(
            vertices=mesh.vertices[atlas["vmapping"]], faces=atlas["indices"],
            vertex_normals=mesh.vertex_normals[atlas["vmapping"]], visual=visual, process=False,
        )

    def _vertex_colored(self, mesh, scene_code):
        query = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(scene_code.device)
        with torch.no_grad():
            colors = self.model.renderer.query_triplane(self.model.decoder, query, scene_code)["color"]
        colored = mesh.copy()
        colored.visual = trimesh.visual.ColorVisuals(colored, vertex_colors=colors.clamp(0, 1).cpu().numpy())
        return colored

    def reconstruct(self, rgba, cfg):
        with torch.no_grad():
            codes = self.model([self._composite(rgba)], device=self.device)
        raw = self.model.extract_mesh(
            codes, False, resolution=self.settings["mc_resolution"], threshold=self.settings["density_threshold"]
        )[0]
        if len(raw.vertices) == 0 or len(raw.faces) == 0:
            raise ImageTo3DError("TripoSR produced an empty mesh", hint="check the background removal image")
        mesh, stats = mesh_post.clean_geometry(raw, cfg)
        if self.settings["bake_texture"]:
            try:
                return Reconstruction(self._bake(mesh, codes[0]), self.name, True, clean_stats=stats)
            except Exception as exc:  # A real mesh with vertex colors is still a valid result
                note = f"texture baking failed, kept vertex colors: {exc}"
                log.warning(note)
                return Reconstruction(self._vertex_colored(mesh, codes[0]), self.name, False, [note], stats)
        return Reconstruction(self._vertex_colored(mesh, codes[0]), self.name, False, clean_stats=stats)


BACKENDS = {TripoSRBackend.name: TripoSRBackend}
_loaded = {}


def _get_backend(name, cfg, device):
    if name not in BACKENDS:
        raise ImageTo3DError(f"unknown 3D model {name!r}", hint=f"choose from {sorted(BACKENDS)}")
    if name not in _loaded:
        _loaded[name] = BACKENDS[name](cfg, device)
    return _loaded[name]


def unload():
    _loaded.clear()


def reconstruct(rgba, cfg, device):
    settings = cfg["image_to_3d"]
    primary, fallback = settings["primary"], settings["fallback"]
    try:
        return _get_backend(primary, cfg, device).reconstruct(rgba, cfg)
    except Exception as exc:
        if not fallback:
            raise
        log.warning("primary 3D model %s failed (%s), trying %s", primary, exc, fallback)
        unload()
        free_gpu_memory()
        result = _get_backend(fallback, cfg, device).reconstruct(rgba, cfg)
        result.notes.append(f"fallback used after {primary} failed: {exc}")
        return result
