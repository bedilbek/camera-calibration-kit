# Kalibr markers for stereo length measurement

**Print any two different IDs at Actual Size / 100%.** The four PDFs contain
100 mm tags that decode as IDs 0, 1, 2, and 3 respectively. IDs 0 and 1 were
the pair used in the two-tag simulation below.

The sequential workflow detects these markers with `DICT_APRILTAG_36h11` and
`markerBorderBits = 2`; the default one-bit setting fails on these Kalibr-style
PDFs. IDs 0 and 1 are matched in both views, and their perspective-corrected centers
are triangulated. This measures center-to-center length.

`tools/evaluate_kalibr_markers.py` renders each PDF and projects it into a
1280 x 720 synthetic camera (`fx = 900 px`) with 3-gray-level Gaussian noise,
0.65-px blur, varied position, and 0-65 degree yaw. Run:

```bash
.venv/bin/python tools/evaluate_kalibr_markers.py > patterns/marker/digital-evaluation.json
```

The checked run decoded PDF IDs `[0, 1, 2, 3]`. At 80 px nominal tag width,
all four PDFs decoded in all 32 trials per angle through 50 degrees; at
65 degrees, 25/32 decoded. At 40 px and 30 degrees, 27/32
decoded; at 20 px, none decoded. Two tags (0 and 1) appeared together in
50/50 trials at 80 px and 30 degrees, with no extra IDs. One hundred simple
negative frames produced no detections. Those counts are controlled simulation
results, not field reliability or a false-positive rate.

For a first print test, keep each tag at least about 80 px wide in **both**
camera images, fully visible, fairly face-on, flat, matte, and still. Estimate
apparent width with `width_px ~= fx_px * 0.1 / distance_m` for a face-on
100 mm tag. This is a starting gate, not an accuracy guarantee. Measure the
printed 100 mm square, test real frames at the intended distances and lighting,
and validate triangulated lengths against a known rigid reference before use.
For a captured image, run
`.venv/bin/python tools/evaluate_kalibr_markers.py --image /path/to/frame.png`;
it prints detected IDs, centers, and corners. Supply `--image` twice to check
one frame from each camera.
