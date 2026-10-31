"""Stage 6. Mesh cleanup and normalization.

clean_geometry runs before texture baking, because decimating afterwards would destroy the UVs.
normalize only moves vertices, so it is safe on a textured mesh.
"""
import numpy as np
import trimesh

from .errors import MeshPostError


def _drop_small_bodies(mesh, min_ratio):
    """Remove floating bits, but keep real separate parts."""
    bodies = mesh.split(only_watertight=False)
    if len(bodies) <= 1:
        return mesh, 0
    largest = max(len(body.faces) for body in bodies)
    kept = [body for body in bodies if len(body.faces) >= min_ratio * largest]
    return trimesh.util.concatenate(kept), len(bodies) - len(kept)


def clean_geometry(mesh, cfg):
    """Takes an uncolored mesh. Returns (clean mesh, stats)."""
    settings = cfg["mesh_post"]
    if len(mesh.faces) == 0:
        raise MeshPostError("the mesh has no faces", hint="check the background removal image")
    stats = {"faces_in": len(mesh.faces)}
    mesh = mesh.copy()
    mesh.merge_vertices(digits_vertex=settings["merge_digits"])
    nondegenerate = mesh.nondegenerate_faces(height=settings["degenerate_height"])
    stats["degenerate_removed"] = int((~nondegenerate).sum())
    mesh.update_faces(nondegenerate)
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh, stats["floaters_removed"] = _drop_small_bodies(mesh, settings["min_body_ratio"])
    target = settings["target_faces"]
    if target and len(mesh.faces) > target:
        mesh = mesh.simplify_quadric_decimation(face_count=target)
    trimesh.repair.fix_normals(mesh, multibody=True)
    if len(mesh.faces) == 0:
        raise MeshPostError("cleanup removed every face", hint="lower mesh_post.min_body_ratio")
    stats["faces_out"] = len(mesh.faces)
    stats["vertices_out"] = len(mesh.vertices)
    stats["watertight"] = bool(mesh.is_watertight)
    return mesh, stats


def normalize(mesh, cfg):
    """Reorder axes to the export convention, center at the origin, scale the largest side."""
    settings = cfg["mesh_post"]
    order = settings["axis_order"]
    normals = np.asarray(mesh.vertex_normals)[:, order]
    vertices = np.asarray(mesh.vertices)[:, order]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    largest = (high - low).max()
    if not np.isfinite(largest) or largest <= 0:
        raise MeshPostError("the mesh has no usable size")
    mesh.vertices = (vertices - (low + high) / 2) * settings["normalize_extent"] / largest
    # Normals computed earlier stay smooth across texture seams, so restore them
    mesh.vertex_normals = normals
    return mesh
