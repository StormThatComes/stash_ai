# Dance contact benchmark

Five photographs show ten dancer instances in supported balances, lifts and overlapping poses. Each JPEG is 500 pixels wide, the scale of a typical VTT thumbnail. See [photo credits and reuse terms](CREDITS.md).

## Run the sweep

From the repository root with the CPU or GPU requirements installed:

```sh
python -m benchmarks.benchmark_dance_contact --device cuda --output dance-contact.json --overlays dance-contact-png
```

Use `--device cpu` for CPU inference. The default `facial_recognition.yml` supplies RF-DETR-XL, PMPose and SAM-pose2seg. Model loading follows the recognition YAML's download policy. Use `--model-dir PATH` to reuse existing weights. The benchmark runs the recognition person backend on each JPEG with no face detections or identity embeddings. Face-guided recovery, scene continuity, and performer assignment are outside the scope of this measurement.

Edit a copy of [profiles.yml](profiles.yml) and pass `--profiles PATH` to choose the values to sweep. Use `--config PATH` to compare against your own baseline. The runner crosses each pose profile with each mask profile. An empty `stock` group inherits the given YAML. Optional PNGs show model keypoints, prompts and masks with the detection tuner's solid/dashed contours. The JSON maps profiles to the numbered PNG directories and reports counts per image.

## What counts as an improvement

The [manual labels](labels.json) are in resized image pixels. Every person has a head center, an approximate full-body box, a subset of visible limb joints, and six points inside their visible silhouette. Each photo has four background points, including gaps near the dancers.

* **People.** One-to-one matching uses detector-box IoU of at least 0.25. Among eligible pairs it maximizes matched count, then combines box IoU and distance to the labelled head center. This helps distinguish partners whose boxes nearly coincide. Misses and unmatched extra detections are reported separately.
* **Pose.** A visible, annotated wrist, elbow, knee or ankle is correct within 8% of that dancer's box diagonal. Matching is one-to-one within each joint type and ignores anatomical left/right. Missing people and missing joints count as misses. This measures sparse limb localization and can miss left/right swaps.
* **Masks.** A clean mask must be available to recognition, cover at least five of the dancer's six points, and cover zero partner or background points. Own-point coverage includes missed people and unavailable masks in its denominator. Partner/background leakage is measured on available, matched masks only. These are sparse ownership checks, so errors between the sampled points can go undetected.
* **Rejected candidates.** For a detected person without an available mask, the report records the production reason. When SAM supplied a candidate, the report scores its cleaned mask (or the raw mask if cleaning never ran) on the same points. A good-looking rejected candidate earns no credit as an available mask.

The fixed tuning set is `attitude`, `penche`, and `paquita`. The ranking maximizes clean masks, then minimizes partner and background hits, maximizes correct joints and matched people, and minimizes extra detections and the number of overrides. Exact ties follow YAML order. Only the stock profile and the tuning winner run on `airborne` and `twisted`. Four photographs share a photographer and several feature the same company. Use this set for repeatable regressions and check promising changes on your own cues.

## CPU reference run

The [24-profile report](../../results/dance_contact_cpu.json) was measured on the repository recognition YAML with RF-DETR-XL 1.11.2, PMPose-B and SAM-pose2seg. The best first-sweep profile raised only `pose.detector.nms_threshold` from 0.70 to 0.85.

| Split and setting | Matched dancers | Correct visible joints | Clean available masks | Extra detections |
| --- | ---: | ---: | ---: | ---: |
| Tune, stock | 4/6 | 14/30 | 3/6 | 0 |
| Tune, NMS 0.85 | 5/6 | 19/30 | 3/6 | 0 |
| Holdout, stock | 4/4 | 13/13 | 1/4 | 0 |
| Holdout, NMS 0.85 | 4/4 | 13/13 | 1/4 | 0 |

On `penche`, higher NMS kept the second dancer. Both poses still reported `no_reliable_prompts`. Reducing `min_prompt_coverage` from 1.0 to 0.66 alone recovered no masks. Raising `max_prompt_points` to 8 with coverage 0.875 lost one clean tuning mask to `low_mask_score`. Allowing foreign-core coverage of 0.15 in that profile did not recover it.

On the holdout, the `airborne` candidate rejected for `foreign_core_in_mask` covered all six own probes, four of six partner probes and a background probe. On `twisted`, a candidate rejected for `excessive_expansion` covered all six own probes and zero partner or background probes. The benchmark checks both cases independently of SAM's own confidence.

### Follow-up for missing reliable prompts

The [second sweep](../../results/dance_contact_prompts_cpu.json) tested ten profiles following the `penche` tuning result. Reproduce it by adding `--profiles benchmarks/data/dance_contact/prompt_profiles.yml` to the command above. Both sweeps chose candidates using the same three tuning photographs and checked them on the same two holdout photographs.

| Change to the shipped YAML | Tune dancers | Tune joints | Tune clean masks | Holdout clean masks |
| --- | ---: | ---: | ---: | ---: |
| Stock | 4/6 | 14/30 | 3/6 | 1/4 |
| Only `pose.refinement.min_prompt_points: 1` | 4/6 | 18/30 | 4/6 | 1/4 |
| Only `pose.detector.nms_threshold: 0.85` | 5/6 | 19/30 | 3/6 | 1/4 |
| Both changes | 5/6 | 19/30 | 3/6 | — |

A single reliable prompt recovered a clean mask on `penche` when the detector returned one dancer. With two detections, the segmenter used different prompts and the combination lost that gain. Lowering only `pose.refinement.min_visibility` to 0.30 admitted a mask covering all six partner probes on the tuning set. The holdout showed identical results for stock and each selected profile. These settings are candidates for cue-level testing. Neither sweep reproduced `missing_prompt_support` or `shared_mask_pixels` on these photographs.

## Calibration on all five examples

Use `--fit-all` to find settings specifically for these photographs. This mode uses all ten annotated dancers for selection, including the four dancers reserved in the earlier sweeps. It requires zero partner hits, zero background hits and zero extra detections, then prefers more clean masks, matched people, correct joints, own hits and fewer overrides, in that order. If every profile fails the three gates, the report records a null winner.

The [finalist settings](calibration_profiles.yml) can be rerun directly with the recognition backend:

```sh
python -m benchmarks.benchmark_dance_contact --fit-all --profiles benchmarks/data/dance_contact/calibration_profiles.yml --device cpu --output dance-calibration.json
```

Add `--overlays dance-calibration-png` to inspect each admitted mask and rejected SAM candidate. The [CPU reference run](../../results/dance_contact_calibration_cpu.json) includes counts and reasons for each photo.

The expanded search scored 1,766 distinct profiles on all five images, reusing image features and identical model inputs. The reference report reruns the shortlisted profiles through the ordinary uncached backend. Both detector and mask parameters were common to all photographs. The trade-off between selecting the most reliable prompts and preserving enough distinct prompts mattered more than lowering the detector score.

| Profile | Matched dancers | Correct limb joints | Clean masks |
| --- | ---: | ---: | ---: |
| Stock | 8/10 | 27/43 | 4/10 |
| Detector NMS 0.94 alone | 10/10 | 38/43 | 5/10 |
| NMS 0.94 and one-or-two prompts | 10/10 | 37/43 | 7/10 |
| NMS 0.94 and four-and-expand | 10/10 | 38/43 | 7/10 |

The final profile uses NMS 0.94, allows one to four positive prompts, requires a 0.65 prompt keypoint score and a 0.45 SAM mask score, and permits union area up to 2.1 times the detector box and side expansion up to 1.1 times the original side. Full positive prompt coverage (1.0), zero allowed foreign-core coverage and duplicate-pose suppression stay enabled. On `attitude`, 0.93 still suppressed the second box while 0.94 kept it. An NMS change alone found both dancers on every photo.

| Photo | Stock clean masks | Final clean masks | Missing masks in the final run |
| --- | ---: | ---: | --- |
| attitude | 1/2 | 1/2 | One pose lacked reliable prompts |
| penche | 0/2 | 1/2 | One pose lacked reliable prompts |
| paquita | 2/2 | 2/2 | None |
| airborne | 0/2 | 1/2 | One candidate contained foreign core points |
| twisted | 1/2 | 2/2 | None |

The seven available masks covered all 42 sampled own points for those dancers, zero of 42 partner probes and zero of 28 background probes. This fit includes the earlier holdout and measures sparse ownership. Its transfer to unrelated VTT cues is untested.
