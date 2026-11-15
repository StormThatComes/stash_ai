# VTT contact holdout

Six 500-pixel thumbnails from two motion pictures in the Library of Congress Edison collection. These photographs are independent of the dance-contact calibration set. Cues 0–2 follow moving dancers seen in contact and from behind. Cues 3–5 show two faces in profile. The edit from cue 2 to cue 3 is a known hard cut. WEBVTT times are intentionally irregular. The reference measures face and pose detection, actual per-person mask admission, continuity and temporal prompt provenance. It does not require a temporal prompt when the current frame already has enough evidence.

- *Buffalo dance* (1894), Edison Manufacturing Co., Library of Congress, [catalogue and original film](https://www.loc.gov/item/00694114/). Three frames from the digital viewing copy starting at 14 seconds, sampled at 2 fps.
- *May Irwin kiss* (1896), Edison Manufacturing Co., Library of Congress, [catalogue and original film](https://www.loc.gov/item/00694131/). Three frames starting at 18 seconds, sampled at 2 fps.

The public-domain film excerpts were rescaled and assembled as one JPEG sprite. Run `python -m benchmarks.benchmark_vtt_contact --model-dir PATH --output report.json` with the recognition dependencies. The command uses SCRFD, the person detector and PMPose/SAM with the ordinary detection audit. Model paths and thresholds come from the selected recognition YAML.

For a second run on these grainy film prints, pass `--face-score-threshold 0.5`. This fixture-only override exercises the face-assisted path. The default run preserves the supplied YAML threshold. Both reports record the effective value and check the known shot cut.
