"""Stage-named errors. A failing stage raises; it never returns a dummy result."""


class Sketch2ShapeError(Exception):
    stage = "pipeline"

    def __init__(self, message, hint=None):
        self.hint = hint
        text = f"[{self.stage}] {message}"
        if hint:
            text += f" | Try: {hint}"
        super().__init__(text)


class ConfigError(Sketch2ShapeError):
    stage = "config"


class DeviceError(Sketch2ShapeError):
    stage = "device"


class WeightsError(Sketch2ShapeError):
    stage = "weights"


class PreprocessError(Sketch2ShapeError):
    stage = "preprocess"


class SketchToImageError(Sketch2ShapeError):
    stage = "sketch_to_image"


class BackgroundError(Sketch2ShapeError):
    stage = "background"


class ImageTo3DError(Sketch2ShapeError):
    stage = "image_to_3d"


class MeshPostError(Sketch2ShapeError):
    stage = "mesh_post"


class ExportError(Sketch2ShapeError):
    stage = "export"


class ValidationError(Sketch2ShapeError):
    stage = "validation"


class MetricsError(Sketch2ShapeError):
    stage = "metrics"
