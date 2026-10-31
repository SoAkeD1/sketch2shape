"""HTML for the app's panels. Presentation only: every value shown comes from a real run."""
import html
import time
from pathlib import Path
from urllib.parse import quote

STAGE_COPY = {
    "preprocess": ("Preprocess", "cleaning your lines"),
    "sketch_understanding": ("Sketch understanding", "guessing what you drew"),
    "sketch_to_image": ("Sketch to image", "painting it in"),
    "background": ("Background removal", "lifting it off the page"),
    "image_to_3d": ("Image to 3D", "building the shape"),
    "mesh_post": ("Mesh cleanup", "tidying the geometry"),
    "export": ("Export", "writing the files"),
    "validation": ("Validation", "checking the files"),
    "metrics": ("Metrics", "measuring the match"),
}
NOTES = (
    "The longest step is image to 3D, usually 15 to 35 seconds.",
    "The back of your object can't be seen, so it's being imagined right now.",
    "The first run after starting takes longer while models load.",
    "A prompt changes color and material, and it can change the shape too.",
    "When it lands, drag to turn it around.",
)
ICON = '<svg class="nb-icon" viewBox="0 0 24 24" aria-hidden="true">{}</svg>'
CHECK = ICON.format('<path d="M5 13l4 4L19 7"/>')
CROSS = ICON.format('<path d="M6 6l12 12M18 6L6 18"/>')
WARN = ICON.format('<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.5"/>')
FILE_ICON = ICON.format('<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v4h4"/>')
CUBE = ICON.format('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>')


def esc(value):
    return html.escape(str(value), quote=True)


def file_url(path):
    return "/gradio_api/file=" + quote(str(path).replace("\\", "/"))


def logo():
    return (
        '<svg class="nb-logo" viewBox="0 0 34 34" aria-hidden="true">'
        '<path d="M5 4h17l7 7v19H5z"/><path d="M22 4v7h7"/>'
        '<path class="t" d="M10.5 18.5l6-3.2 6 3.2v6.6l-6 3.2-6-3.2z"/>'
        '<path class="t" d="M10.5 18.5l6 3.2 6-3.2M16.5 21.7v6.6"/>'
        '<circle cx="25.5" cy="14.5" r="2.4"/></svg>'
    )


def header():
    return (
        f'<header class="nb-header">{logo()}'
        '<span class="nb-word">Sketch<span class="nb-mono nb-two">2</span>Shape</span>'
        '<span class="nb-mono nb-crumb">one sketch &rarr; one 3d model</span>'
        '<button class="nb-theme" type="button" aria-label="Switch light or dark theme" onclick="window.nbToggleTheme()">'
        + ICON.format('<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>')
        + "</button></header>"
        '<div class="nb-hero"><h1>Draw one object. Get a <span class="nb-hl">3D model</span> you can turn around.</h1>'
        "<p>Watch your own lines become a textured model. No 3D skills needed.</p></div>"
        '<div class="nb-tip" id="nb-tip" hidden><span>First time? The page is ready, so just start drawing. '
        "A prompt is optional: leave it empty and it will guess what you drew.</span>"
        '<button type="button" onclick="window.nbDismissTip()">Got it</button></div>'
    )


def empty_panel():
    return (
        '<div class="nb-empty"><svg viewBox="0 0 220 150" aria-hidden="true">'
        '<path class="g" d="M20 128h180"/>'
        '<path class="d" d="M60 40c0-9 22-15 50-15s50 6 50 15-22 15-50 15-50-6-50-15zM60 40l8 72c0 8 18 14 42 14s42-6 42-14l8-72"/>'
        '<path class="d" d="M160 60c20 0 28 12 28 26s-10 28-32 28"/>'
        '<circle cx="186" cy="30" r="5"/></svg>'
        '<span class="nb-mono nb-eyebrow">3d preview</span>'
        "<strong>Your model will land here</strong>"
        "<span>Draw on the page and press Generate.</span></div>"
    )


class RunTracker:
    """Turns stage events into the live panel. Only reports what the pipeline actually says."""

    def __init__(self, stages, has_prompt, max_attempts, variant_total=None):
        self.stages = [s for s in stages if not (has_prompt and s == "sketch_understanding")]
        self.max_attempts = max_attempts
        self.variant_total = variant_total
        self.variant = 1
        self.started = time.time()
        self.reset()

    def reset(self):
        self.times, self.current, self.current_since = {}, None, None
        self.attempt, self.turning = 1, False
        self.images = {}

    def begin_variant(self, number):
        if number != self.variant:
            self.variant = number
            self.reset()

    def stage(self, name):
        now = time.time()
        if self.current:
            self.times[self.current] = self.times.get(self.current, 0) + now - self.current_since
        self.turning = name == "sketch_to_image" and name in self.times
        if self.turning:
            self.attempt += 1
            for later in self.stages[self.stages.index(name):]:
                self.times.pop(later, None)
            for stem in ("04_generated", "05_foreground"):
                self.images.pop(stem, None)
        self.current, self.current_since = name, now

    def image(self, stem, path):
        self.images[stem] = path

    def _layer(self, stem, css):
        path = self.images.get(stem)
        return f'<img class="{css}" src="{esc(file_url(path))}" alt="">' if path else ""

    def render(self):
        total = len(self.stages)
        current = self.current or self.stages[0]
        index = self.stages.index(current)
        name, doing = STAGE_COPY[current]
        rows = []
        for number, stage in enumerate(self.stages, start=1):
            title, _ = STAGE_COPY[stage]
            if stage == current:
                status, right = "running", f'<span class="nb-elapsed" data-since="{self.current_since or time.time()}"></span>'
            elif stage in self.times:
                status, right = "done", f"{self.times[stage]:.1f} s"
            else:
                status, right = "waiting", "waiting"
            mark = CHECK if status == "done" else str(number)
            rows.append(
                f'<li class="nb-stage is-{status}"><span class="nb-num nb-mono">{mark}</span>'
                f'<span class="nb-stage-name">{esc(title)}</span><span class="nb-mono nb-time">{right}</span></li>'
            )
        retry = ""
        if self.attempt > 1:
            retry = (
                f'<div class="nb-retry"><span class="nb-mono">quality check · attempt {self.attempt} of {self.max_attempts}</span>'
                "That version drifted from your lines, so it's turning the page and trying another seed.</div>"
            )
        eyebrow = f"stage {index + 1} of {total} · {esc(doing)}"
        if self.variant_total:
            eyebrow = f"variant {self.variant} of {self.variant_total} · " + eyebrow
        notes = "".join(f'<span style="animation-delay:{i * 6}s">{esc(n)}</span>' for i, n in enumerate(NOTES))
        sheet = "".join((
            '<div class="nb-paper"></div>',
            f'<div class="nb-soak">{self._layer("03_sketch_mask", "nb-soak-mask")}</div>',
            self._layer("01_clean_sketch", "nb-lines"),
            self._layer("04_generated", "nb-gen"),
            self._layer("05_foreground", "nb-fg nb-slice s1"),
            self._layer("05_foreground", "nb-fg nb-slice s2"),
            self._layer("05_foreground", "nb-fg nb-cut"),
        ))
        turn = " nb-turn" if self.turning else ""
        return (
            f'<div class="nb-run"><div class="nb-sr" aria-live="polite">Stage {index + 1} of {total}: {esc(name)}.</div>'
            f'<span class="nb-mono nb-eyebrow">{eyebrow}</span>'
            "<h2>Your sketch is becoming an object</h2>"
            f'<div class="nb-sheet f-{current}{turn}"><div class="nb-lift">{sheet}</div><div class="nb-shadow"></div></div>'
            f'<div class="nb-note-roll nb-mono">{notes}</div>{retry}'
            f'<ol class="nb-stages">{"".join(rows)}</ol></div>'
        )


def score_badge(meta):
    scores = meta["metrics"]
    pct = round(scores["mesh_ink_coverage"] * 100)
    warnings = "".join(f'<div class="nb-warn">{WARN}<span>{esc(w)}</span></div>' for w in meta["warnings"])
    prompt_note = "A loose match this time. Try 3 variants, or add a short prompt." if pct < 70 else ""
    low = f'<p class="nb-low">{esc(prompt_note)}</p>' if prompt_note else ""
    return (
        '<div class="nb-badge nb-rise"><span class="nb-mono nb-eyebrow">sketch match</span>'
        f'<h2><span class="nb-hl nb-mono" data-count="{pct}">{pct}%</span> of your strokes are on the 3D model</h2>'
        f'<p class="nb-mono nb-sub">silhouette IoU {scores["silhouette_iou"]:.2f} · seed {esc(scores["seed"])} · {meta["total_seconds"]:.0f} s</p>'
        f"{low}{warnings}</div>"
    )


def angle_strip(views_path, count):
    url = esc(file_url(views_path))
    cells = []
    for index in range(count):
        offset = index * 100 / max(1, count - 1)
        cells.append(
            f'<span class="nb-angle" role="img" aria-label="Model at {round(index * 360 / count)} degrees" '
            f'style="background-image:url(\'{url}\');background-size:{count * 100}% 100%;background-position:{offset}% 0"></span>'
        )
    return (
        f'<div class="nb-angles"><span class="nb-mono nb-eyebrow">{count} renders around the model</span>'
        f'<div class="nb-angle-row" style="grid-template-columns:repeat({count},minmax(0,1fr))">{"".join(cells)}</div>'
        '<p class="nb-guess"><span class="nb-mono">this part is a guess</span> '
        "The back and underside were imagined from one front view. Turn it around before you rely on them.</p></div>"
    )


def match_panel(run_id, files):
    """Sketch, model and overlay views. Plain radio inputs, so switching needs no server call."""
    group = "nbm" + esc(run_id).replace("-", "")
    views = (("Overlay", "07_sketch_match"), ("Sketch", "01_clean_sketch"), ("Generated", "04_generated"))
    parts = ['<div class="nb-match nb-rise"><div class="nb-seg" role="radiogroup" aria-label="Sketch match view">']
    panes = []
    for index, (label, stem) in enumerate(views):
        checked = " checked" if index == 0 else ""
        parts.append(
            f'<input type="radio" name="{group}" id="{group}{index}" class="nb-r{index}"{checked}>'
            f'<label for="{group}{index}">{label}</label>'
        )
        panes.append(f'<img class="nb-pane nb-p{index}" src="{esc(file_url(files[stem]))}" alt="{label} view">')
    parts.append('<div class="nb-panes">' + "".join(panes) + "</div></div>")
    parts.append(
        '<div class="nb-legend"><span><i class="on"></i>Solid teal: on the model</span>'
        '<span><i class="off"></i>Dashed red: missed</span></div>'
        '<p class="nb-explain">This measures how much of your line lands on the model\'s front view, not how good it looks.</p></div>'
    )
    return "".join(parts)


def _size(path):
    return f"{Path(path).stat().st_size / 2 ** 20:.1f} MB"


def downloads_panel(files, zip_path):
    cards = []
    if "glb" in files:
        cards.append(
            f'<a class="nb-dl primary" href="{esc(file_url(files["glb"]))}" download="model.glb">{CUBE}'
            '<span><strong>Download .glb</strong><small>One file, texture included. Best choice.</small></span>'
            f'<span class="nb-mono">{_size(files["glb"])}</span></a>'
        )
    if zip_path:
        cards.append(
            f'<a class="nb-dl" href="{esc(file_url(zip_path))}" download="model_obj.zip">{FILE_ICON}'
            "<span><strong>Download .obj set (.zip)</strong><small>.obj, .mtl and texture, kept together.</small></span>"
            f'<span class="nb-mono">{_size(zip_path)}</span></a>'
        )
    return (
        '<div class="nb-downloads nb-rise"><span class="nb-mono nb-eyebrow">take it with you</span>'
        + "".join(cards)
        + '<p class="nb-explain">Opens in Blender and other 3D software.</p></div>'
    )


def _rows(items):
    rows = "".join(
        f'<div class="nb-row"><span class="nb-mono nb-val">{esc(value)}</span>'
        f"<div><strong>{esc(name)}</strong><small>{esc(why)}</small></div></div>"
        for value, name, why in items
    )
    return f'<div class="nb-rows">{rows}</div>'


def quality_panel(meta):
    scores = meta["metrics"]
    items = [
        (f"{scores['mesh_ink_coverage']:.0%}", "Sketch match", "Share of your strokes on the model's front view."),
        (f"{scores['ink_coverage']:.0%}", "Ink coverage", "Share of your strokes on the generated image. Used to decide retries."),
        (f"{scores['silhouette_iou']:.2f}", "Silhouette IoU", "Overlap of the model's outline with your filled sketch. Low on open or broken sketches."),
        (f"{scores['clip_sketch_to_views']:.2f}", "CLIP sketch to views", "How alike the sketch and the model's renders look to CLIP."),
        (f"{scores['clip_view_consistency']:.2f}", "CLIP view consistency", "Whether all sides look like one object."),
    ]
    if "clip_text_to_views" in scores:
        items.append((f"{scores['clip_text_to_views']:.2f}", "CLIP text to views", "How well the renders match the text used. Raw similarity, not a percentage."))
    if meta.get("subject_confidence") is not None:
        items.append((f"{meta['subject_confidence']:.2f}", f"CLIP read: {meta['subject'] or 'not sure'}", "How sure CLIP was about what you drew. Used only with no prompt."))
    return _rows(items)


def performance_panel(meta):
    totals = {}
    for stage in meta["stages"]:
        totals[stage["stage"]] = totals.get(stage["stage"], 0) + stage["seconds"]
    longest = max(totals.values(), default=1) or 1
    bars = "".join(
        f'<div class="nb-bar"><span>{esc(STAGE_COPY.get(name, (name,))[0])}</span>'
        f'<div class="nb-track"><i style="width:{max(1, seconds * 100 / longest):.1f}%"></i></div>'
        f'<span class="nb-mono">{seconds:.1f} s</span></div>'
        for name, seconds in totals.items()
    )
    tiles = "".join(
        f'<div class="nb-tile"><span class="nb-mono">{label}</span><strong class="nb-mono">{value}</strong></div>'
        for label, value in (
            ("total", f"{meta['total_seconds']:.1f} s"),
            ("peak gpu memory", f"{meta['peak_vram_gb']:.2f} GB"),
            ("peak ram", f"{meta['peak_ram_gb']:.1f} GB"),
        )
    )
    return f'<div class="nb-bars">{bars}</div><div class="nb-tiles">{tiles}</div>'


def checks_panel(meta):
    attempts = "".join(
        f'<li><span class="nb-num nb-mono">{esc(a["attempt"])}</span><span>seed {esc(a.get("seed", "n/a"))} · '
        f'{esc(a["outcome"])}</span></li>'
        for a in meta["attempts"]
    )
    chips = []
    for fmt, report in meta["validation"].items():
        chips.append(f'<span class="nb-chip">{CHECK}.{esc(fmt)} valid · {esc(report.get("size_mb"))} MB</span>')
        chips.append(f'<span class="nb-chip">{CHECK}.{esc(fmt)} {"texture" if report.get("textured") else "vertex colors"} kept</span>')
    return f'<ol class="nb-attempts">{attempts}</ol><div class="nb-chips">{"".join(chips)}</div>'


def geometry_panel(meta):
    scores = meta["metrics"]
    tiles = (
        ("faces", f"{scores['faces']:,}", ""),
        ("watertight", "yes" if scores["watertight"] else "no", "Printing may need a repair step." if not scores["watertight"] else ""),
        ("floaters removed", scores["floaters_removed"], "Stray bits of mesh cleaned away."),
        ("box aspect ratio", f"{scores['bbox_aspect_ratio']:.2f}", "Thinnest side divided by the largest."),
    )
    cells = "".join(
        f'<div class="nb-tile"><span class="nb-mono">{label}</span><strong class="nb-mono">{esc(value)}</strong><small>{esc(why)}</small></div>'
        for label, value, why in tiles
    )
    return f'<div class="nb-tiles">{cells}</div>'
