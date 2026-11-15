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

The fixture `candy-balance` also reproduces a real `missing_prompt_support` rejection. Starting from stock, set `pose.refinement` to one through four prompts, `min_keypoint_score` to 0.6, `min_visibility` to 0.3 and `min_mask_score` to 0.3. The rejected cleaned candidate contained three of eight target probes and all eight partner probes. The per-photo report distinguishes such a candidate from a usable recognition mask.

## CPU results

The [per-photo CPU report](../../results/dance_contact_cpu.json) was measured with RF-DETR-XL, PMPose and SAM-pose2seg. The same settings were used on every photograph.

| Profile | People matched / 34 | Clean masks / 34 | Photos with a clean mask / 17 | All people clean / 17 photos | Correct limb joints / 138 | Partner / background hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock` | 32 | 20 | 13 | 9 | 112 | 0 / 0 |
| `four-prompts` | 34 | 21 | 15 | 8 | 119 | 5 / 0 |
| `single-prompt` | 34 | 27 | 14 | 12 | 124 | 0 / 2 |
| `common` | 34 | 26 | 16 | 10 | 124 | 0 / 0 |

The search varied prompt count, keypoint confidence, visibility, presence, neighbour overlap, detector thresholds, expected SAM mask quality, positive/foreign coverage and expansion limits. For gate sweeps, identical model inputs were reused. Each distinct admission outcome for a given set of candidate masks was scored with the recognition path, including mask conflicts and the masked second pose pass. The shortlisted settings were checked using the ordinary backend.

### Applying the common profile

Relative to `single-prompt`, `common` changes two effective thresholds. It raises the prompt keypoint score from 0.50 to 0.75 and lowers the SAM mask score from 0.50 to 0.45. The foreign-core allowance remains zero, and full positive-prompt coverage is required.

Merge the following leaves into the existing `pose` section of a recognition YAML. [profiles.yml](profiles.yml) contains the equivalent dotted overrides used by the benchmark.

```yaml
pose:
  detector:
    nms_threshold: 0.94
  refinement:
    max_prompt_points: 1
    min_keypoint_score: 0.75
    min_prompt_points: 1
    min_mask_score: 0.45
    max_box_area_ratio: 3.0
    max_side_expansion: 1.1
```

| Photograph | People | Stock clean masks | Common clean masks | Missing mask reasons with common settings |
| --- | ---: | ---: | ---: | --- |
| `airborne` | 2 | 0 | 1 | 1 × `no_reliable_prompts` |
| `attitude` | 2 | 1 | 1 | 1 × `no_reliable_prompts` |
| `beach-solo` | 1 | 1 | 1 | — |
| `candy-balance` | 2 | 0 | 0 | 2 × `no_reliable_prompts` |
| `floor-reach` | 2 | 2 | 2 | — |
| `floor-trio` | 3 | 0 | 2 | 1 × `no_reliable_prompts` |
| `low-spiral` | 2 | 2 | 2 | — |
| `paired-arms` | 2 | 2 | 2 | — |
| `paquita` | 2 | 2 | 2 | — |
| `parallel-jump` | 2 | 2 | 2 | — |
| `penche` | 2 | 0 | 1 | 1 × `no_reliable_prompts` |
| `pointe-rear` | 2 | 2 | 1 | 1 × `no_reliable_prompts` |
| `silhouette-solo` | 1 | 1 | 1 | — |
| `spiral-entry` | 2 | 1 | 1 | 1 × `no_reliable_prompts` |
| `spiral-side` | 2 | 2 | 2 | — |
| `stage-trio` | 3 | 1 | 3 | — |
| `twisted` | 2 | 1 | 2 | — |

On `candy-balance` a further 2,016 detector/prompt combinations yielded no clean mask. One frequent rejected candidate covered all eight target probes and all eight partner probes. Loosening the foreign-core threshold to accept that candidate would merge the two people in the mask.

The report also counts predicted joints near a labelled partner joint when every annotated target joint of that type is outside tolerance. Eleven manually sided wrist landmarks provide a separate anatomical check. A sided landmark counts as correct or swapped only when exactly one predicted wrist is within tolerance. Reported runtime includes model loading. Encoder and decoder counts include initial and recovery forwards.
