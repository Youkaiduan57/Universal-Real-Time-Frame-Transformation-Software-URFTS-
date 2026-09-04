# Upscaling without frame generation

## Experimental reuse of unchanged regions

Enable **AI Settings → Reuse unchanged tiles** to retain previous AI results for
unchanged image regions. The first frame is fully processed. Subsequent frames
compare every source pixel, including overlapping borders, and infer only changed
tiles. Completely unchanged frames also skip tile recomposition. Cached tiles
refresh after 120 reuses; resizing and processor shutdown discard the cache.

This option forces small tiles (normally at most 256 pixels) even with tiling Off
or Auto. It preserves the selected source resolution and existing overlap blending.
It uses additional memory for source and output caches and is disabled by default.
It is intended for spatial image models, not stateful temporal ONNX models.
Logs report inferred and reused tile counts every five seconds during processing.

Camera movement, compression noise, animated water and lighting changes can invalidate
most tiles. In those cases it may be slower than ordinary full-frame inference.
It is not optical-flow tracking or an FPS guarantee. Compare moving and stationary
segments, and disable it if most tiles are continually inferred. AI quality and
seams remain subject to the chosen model's existing tiled-inference limitations.

Changed tiles now recompose only their output rectangles, including neighbouring
overlap contributions. The completed output is retained; returning an independent
image still requires a full output copy. `AI cached stages` logs separate model
inference, dirty-region composition and total processor time (not presentation).
With AI reuse enabled and FG off, an idle WGC timeout retains the previous image
instead of recreating capture. First-frame failures, minimized/closed windows and
runtime errors still use normal error handling. Idle images are repeats, not new
video frames; do not interpret their presentation rate as unique source FPS.

In the Upscaling card select **Workflow → Upscaling only — no generated frames**.
The frame-generation card is disabled and no interpolation model is initialized.
Switching back restores the saved frame-generation selection. Save separate GUI
profiles for gaming, calls, and media instead of replacing the gaming baseline.

## Responsive viewing: cloud gaming and calls

- CPU-frame pipeline, WGC, target the application window.
- Shader, Bicubic, refinement Off or Subtle.
- FPS Auto, presentation buffer 0 ms, queue depth 2.

This enlarges and optionally sharpens the received picture. It does not increase
the source stream bitrate, restore guaranteed original detail, or generate frames.
For calls it changes your local view only, not the video sent to other participants.
Protected/DRM video can block capture; this app does not bypass that protection.

## AI quality evaluation: media or a stationary scene

- Upscaling only, CPU-frame pipeline, AI, DirectML.
- **AI input → Native source — preserve input detail**, tiling Auto.
- Refinement Off initially, presentation buffer 0 ms.

Native source passes the original captured pixels to the existing SRVGG x2 model
without the old preset resize. A 1280×720 source therefore produces a 2560×1440
model output before the existing presentation resize; this is not a native x1.5
model. It can be substantially slower than the former reduced-input path.
Tiling limits working memory but does not remove the computation for those pixels.
The existing model is not newly trained for games, faces, text, or video temporal
consistency. Inspect all of these before adopting it. No real-time FPS promise is
made for full-source AI inference, especially on an iGPU.

**Custom size** retains manual input width/height for explicit speed experiments.
Changing the performance/FG preset no longer overwrites AI dimensions or tiling.
Existing settings migrate as Custom to avoid silently increasing GPU load; select
Native explicitly when making a new quality-test profile.

Compare a stationary scene with FG off and identical output dimensions. Then play
30 seconds of moving content and collect logs, checking halos, temporal shimmer,
text/faces, latency and sustained FPS. The next quality milestone is to evaluate
a faster video-suitable SR model on those measurements, not assume that larger
input or more sharpening automatically improves usable quality.
