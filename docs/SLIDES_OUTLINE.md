# Slides outline

Eight slides. Rule-book weighting: Technical Accuracy 50%, Innovation 20%, Execution 15%, Presentation 15%.

1. **Problem**
   - 3D modeling needs expert skills; a sketch is where ideas start.
   - One sketch lacks depth and hidden surfaces.
2. **Pipeline** (flowchart from `README.md`)
   - Sketch → understand → image → isolate → check → 3D → validate.
3. **Live demo**
   - Judge draws a fresh sketch, no prompt.
   - Generate, rotate, download, open Details.
   - Have the app started before this slide: startup takes about 60 s.
4. **Innovation**
   - CLIP names the sketch when there's no prompt.
   - Ink coverage gate: retries before the slow 3D stage.
   - Sketch match view: green strokes on the model, red where they missed.
   - Generate 3 variants: ranks its own results. Table example: top result had four legs, while the default seed had two slabs.
   - Runs on an AMD GPU despite NVIDIA-focused tooling.
5. **Metrics** (`docs/METRICS.md`)
   - 18 of 18 valid, ink coverage 0.97, 31 s, 2.68 GB VRAM.
   - Flower example: coverage 0.47 → 0.96 after the fix.
   - State the test set: 2 distinct user sketches plus variations.
6. **Limitations**
   - Hidden geometry is a guess.
   - Table's legs example.
   - Tested end to end on one GPU.
7. **Applications**
   - Product design, architecture, games, education.
8. **Disclosure**
   - Models, licenses, no custom dataset.
   - AI assistant: Claude Code (Anthropic), what for and why, from `ATTRIBUTION.md`.

**Backup for the demo:** record a screen video of one full run, in case the live demo fails.
