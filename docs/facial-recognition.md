# Facial recognition

Recognition is an explicit VTT-based workflow, independent of tagging. It creates
anonymous Stash performers and additive scene links from retained face/body evidence.
Optional video decoding is used only for portraits. It is never started by `--models all`.

## Run recognition

Install the [CPU or GPU application profile](../constraints/README.md), then the extras
used by the shipped RF-DETR-XL and ViTPose++ Large configuration:

```powershell
python -m pip install -r requirements-sapiensid.txt -r requirements-rfdetr.txt
python stash_ai.py recognition run --scene-id 123
python stash_ai.py recognition run --scene-id 123 --scene-id 456
python stash_ai.py recognition run --all-scenes
```

Set `STASH_URL` and, when required, `STASH_API_KEY` as in the [quick start](../README.md#quick-start).
`--all-scenes` snapshots the scene IDs for one finite invocation. Missing pinned
assets download on first use when `execution.models.download_missing` is enabled.
Use `--face-config PATH` and `--faces-db PATH` to select configuration and state.
CUDA is the default; use `--device cpu` with a CPU installation. Only explicit
`auto` permits CPU fallback. See [model/runtime setup](recognition/models.md).

## Choose a workflow

| Task | Guide |
| --- | --- |
| Inspect detector output or compare one cue across settings | [Detection audits](recognition/detection.md) |
| Select face recognizer, person detector or pose model | [Face models](recognition/models.md) · [Pose backends](recognition/pose-backends.md) |
| Configure body evidence and face/body merge rules | [SapiensID](sapiensid-hybrid.md) |
| Understand anchors, auxiliary samples and views | [Identity matching](recognition/identity-matching.md) · [Reliability rules](recognition/reliability.md) |
| Refresh portraits, change crop policy or apply names | [Portraits and names](recognition/portraits.md) |
| Export identity evidence or inspect fragmentation | [Reports and exports](recognition/reports.md) |
| Tune parameters against explicit performer sets | [Solver](recognition/solver.md) · [JSON report](recognition/solver-report.md) |
| Run concurrent solver trials | [Parallel solver](recognition/solver-parallel.md) |
| Inspect, prune or compact the recognition database | [State and maintenance](recognition/storage.md) |
| Diagnose slow runs, memory or recovery | [Execution](recognition/performance.md) · [Portrait performance](recognition/portrait-performance.md) |
| Measure correctness and speed | [Evaluation](recognition/evaluation.md) · [Benchmarks](recognition/benchmarking.md) |

## Recognition YAML v2

The shipped [facial_recognition.yml](../facial_recognition.yml) lists every
setting with operator help immediately below it. The file groups settings by
stage: face scanning, pose detection, SapiensID collection, identity matching,
portraits, execution, storage and naming. Recognition runs use RF-DETR-XL for
person boxes and ViTPose++ Large for COCO17 points. Body evidence collection
is enabled, including VTT cues without an accepted face. Creating body-only
identities and global body merging have their own switches under `matching.body`.

Existing version 1 files still use their original settings and omitted-field
defaults. To convert a local file and retain its effective policy, run:

```shell
python stash_ai.py recognition migrate-config --face-config local.yml --output local-v2.yml
```

The output must be a new file. This offline command needs neither the recognition
database nor model weights. Keeping the files in the same directory preserves
relative paths. Exporting elsewhere rebases paths to their original locations.
Equivalent settings reuse the face scan, person evidence and body contracts.
Changing the actual model or a collection threshold still changes its contract.

| In a v1 file | In a v2 file |
| --- | --- |
| `models.detector`, `models.recognizer` | `face.detector.backend`, `face.recognizer.backend` |
| `sapiensid.pose.detector`, `sapiensid.pose.backend` | `pose.detector.backend`, `pose.estimator.backend` |
| `sapiensid.pose.batch_size` | `pose.detector.frame_batch_size` and `pose.estimator.vitpose_plus_large.batch_size` |
| `clustering.min_auxiliary_face_pixels`, `clustering.min_anchor_face_pixels` | `matching.face.samples.minimum.min_face_pixels`, `matching.face.anchors.quality.min_face_pixels` |
| `sapiensid.identities.anchor_keypoint_rules` | `matching.body.anchors.keypoint_rules` |
| `sapiensid.matching.face_floor` | `matching.hybrid.face_floor` |
| `execution.rebuild_memory_bytes` | `execution.rebuild.memory_bytes` |

Detection sweeps and tuning grids accept either spelling. Solver winners and
YAML diffs use the layout of their input. A diff is applied to that input.
Under `pose.point_quality`, set the shared PMPose thresholds for SAM refinement
and duplicate detection. A stage may override any threshold; `null` inherits it.
The final pose-estimator filter and body-anchor rules are independent.

Recognition configuration examples in these guides use v2. Merge example fields
into the corresponding sections of your file; omitted fields retain your settings.
Existing diagnostic reports may still use internal v1 names, and solver output
follows its input layout. The separate performer-names file retains its own version 1 schema.

## State and publication

Ownership uses an exact installation/person token in performer details, never a
name-only match. Each mutation intent is committed to the face SQLite before
dispatch. Lost creation/link/unlink and legacy merge responses are reconciled by fresh Stash reads;
unresolved outcomes are withheld without blindly retrying. Existing manual links
are preserved with Stash's `ADD` mutation. Selected-scene runs do not resume links
for other scenes. Exit 0 means the finite selection completed; exit 1 records
partial/failed work, and interruption returns 130 from the root launcher.

`facial_recognition.yml` configures face and body evidence, matching and portraits. Numeric matching
thresholds are cosine similarities, not calibrated probabilities. The initial
0.65 match/0.60 cohesion defaults prioritize withholding uncertainty; they do not
claim measured accuracy on a user's VTT library. OpenCV's example LFW threshold
is not used as a Stash-thumbnail accuracy guarantee.

All relative model, database and names paths resolve against the selected YAML
directory. Missing default YAML uses bundled defaults; an explicitly selected
missing file fails. Values are immutable during an invocation. The default
`faces_state.sqlite3` is reserved for face information and is independent of
tagger state/history. A names/portrait/threshold setting is not a raw model identity.
