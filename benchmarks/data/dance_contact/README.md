# Dance contact benchmark

One recognition configuration runs on all 17 annotated photographs (34 people). The fixtures cover overlapping duets, floor contact, lifts, cropped bodies, two solos and two trios. Each image is 500 pixels wide. [Credits, sources and licenses](CREDITS.md) accompany every image.

## Run

From the repository root with the recognition dependencies and pinned model weights:

```sh
python -m benchmarks.benchmark_dance_contact --device cpu --output dance-contact.json
```

Use `--device cuda` for GPU inference, `--model-dir PATH` for an existing weight directory, `--config PATH` for another recognition YAML, or `--overlays DIRECTORY` for detection-tuner overlays. The runner uses recognition's RF-DETR-XL, PMPose and SAM-pose2seg person path. It supplies no face detections or identity embeddings.

[profiles.yml](profiles.yml) lists the public YAML overrides for each profile. The empty `stock` profile inherits the supplied recognition configuration. Use `--profiles PATH` for a custom comparison. A two-axis `pose` × `mask` sweep file is also accepted. Every profile runs on the same complete set. The JSON reports coverage, per-photo counts, rejection reasons and sparse ownership scores for rejected candidates.

## Scoring

* **Detection.** One-to-one assignment requires detector-box IoU of at least 0.25 and uses the annotated head center to distinguish overlapping people when visible. Misses and extra detections are counted separately.
* **Pose.** An annotated wrist, elbow, knee or ankle matches within 8% of the person's box diagonal. Assignment is one-to-one within each joint type and ignores anatomical left/right. Missed people contribute zero correct joints.
* **Mask ownership.** A clean mask must be available to recognition, cover at least 75% of its own foreground probes, and contain zero partner or background probes. This requires five of six or six of eight own probes depending on the annotation. Missing masks receive zero credit. Sparse points cannot detect every boundary error. A rejected SAM candidate is scored separately when available.
* **Selecting a common profile.** Eligible profiles have zero partner hits, background hits and extra detections across the entire benchmark. Compare each profile's clean-mask fractions for the 17 photos, ordered from worst to best. The first differing fraction decides. Remaining ties use total clean masks, matched people, correct joints, own probe hits and fewer overrides. A null winner means no profile met the ownership and extra-detection gates.

All 17 photos participate in calibration. These results measure performance on these examples. Independent validation requires new photos or VTT cues.
