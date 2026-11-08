# Measuring recognition identity rebuild

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

Rebuild consumes retained embeddings. Its benchmark does not load recognition
models, fetch media, contact Stash or publish assignments. GPU inactivity during
this stage is expected. The serial default remains `rebuild_workers: 1`.

Start with a representative selection and the same facial configuration used by
the run. The local benchmark opens the source SQLite read-only, takes a consistent
snapshot, and closes the source before measuring. It copies recognition evidence
and historical labels in the selected scope/namespace into RAM; it does not read
JPEG crops, portrait images, quality BLOBs or run history. Every worker-budget
trial starts from the same private snapshot. Disk replay checkpoints are disabled
for this command, regardless of the configuration.

```powershell
python -m benchmarks.benchmark_recognition_retained --config facial_recognition.yml --scene-id 100 101 102 --workers 1 8 16 20 --repeats 3 > rebuild-baseline.json
```

Use `--all-scenes` instead of `--scene-id` for the full retained library. Choose
`--database` to override the configured source, and `--scope` if the namespace is
present in multiple Stash scopes. `--namespace` is an explicit override for older
retained evidence; normally the recognizer namespace comes from the configuration.
An explicitly selected scene needs its latest scan to be complete. All-scenes
mode excludes latest incomplete scans and reports their count; it never silently
substitutes an older complete scan.

The snapshot retains historical references needed for stable identity binding,
including observations outside the selected scenes. This preserves the real
rebuild's input semantics. RAM can therefore exceed just the selected embeddings,
and several RAM database copies coexist during a trial. Snapshot time is reported
separately from cold/cached rebuild wall time. No disk-backed working cache is
created. The JSON output itself is a report that the shell can save to a file.

Output contains input/configuration/selection hashes, counts, dimensions,
environment, effective worker/native limits, cold and cached timings, process
CPU seconds and semantic-output hashes. It omits scene/person IDs, URLs,
installation IDs, embeddings and SQL. Nothing is uploaded. Treat hashes and
workload counts as local diagnostics and review any report before sharing it.

Memory uses a 50 ms process-RSS sampler. `sampled_peak_bytes` can miss shorter
peaks and includes the interpreter, source snapshot and private trial/replay.
`rss_after_bytes` is current end-of-trial RSS, not a peak or retained-buffer cap.
Compare trials in the same process/interpreter and report the sampling method.

## Interpreting CPU diagnostics

The console's CPU section shows active/configured workers, computed/retired
blocks, admitted queue bytes, coordinator wait reason and strict scenes applied.
Workers update only lock-protected execution counters. The coordinator owns all
console updates. Computed blocks never advance the scene bar or its ETA.

With `profile_rebuild: true`, `rebuild_profile.cpu` includes:

- `completed_blocks`, `failed_blocks`, `active_workers`, `pending_blocks` and
  `retired_blocks`. Legacy `applied_blocks` also means retired numerical blocks;
  it does not mean identity decisions or a committed durable rebuild.
- `strict_scenes_applied`, counting completed strict replay scenes. Checkpoint
  reuse and final durable publication are separate stages.
- Bounded `fallbacks` for disabled parallelism, small catalogues/single shards,
  unsupported layouts, mixed dimensions and admission limits. A fallback does
  not suppress the original retained-evidence validation error.
- `preparation` time for vectors and packed matrices; profile counters distinguish
  `repacked_bytes` and `reused_packed_bytes`.
- `records`, separating computed, consumed and discarded score rows, evidence
  rows and constituent proofs. Consumed means delivered to the canonical
  numerical consumer, not an accepted identity. An early exit can leave valid
  speculative answers unused.

Worker wall seconds are summed task durations. Worker-thread CPU excludes native
helper threads. Neither metric is elapsed run time, CPU utilization or ETA.
Preparation and other stage timers can nest; do not sum them as disjoint shares.

Measure a baseline now and repeat representative small, merge-heavy and full
selections after major CPU changes. For AuraFace, prioritize 512-dimensional
workloads. Compare 1/8/16/20 workers with native limits recorded in the report;
the largest count is not presumed fastest. Different hashes between budgets are
a correctness failure: the command exits unsuccessfully.

The synthetic `benchmark_recognition_parallel` remains useful for offline smoke
tests, but its orthogonal repeated identities do not predict a real 9,000-scene
library. Representative 24-core measurements and a measured tuning/default
recommendation remain tracked in #856.

### Crowded VTT pairs

Run `python -m benchmarks.benchmark_crowded_vtt --config facial_recognition.yml --labels pairs.json --cuda-profile`
against Stash using its usual connection settings. Each profile scans the same complete VTT scenes into a disposable
database and rebuilds identities locally. The profiles are unmasked PMPose, SAM on overlapping boxes, and SAM with
foreground SapiensID crops. The benchmark needs the models on the machine performing the scan. CUDA profiling is
optional and reports summed CUDA activity time separately from scan wall time.

Labels map scene and zero-based cue to every visible performer and their torso centre in original VTT pixels.
Mark separate clear cues as references for the same performer. For example:

```json
{"42":{"0":[{"truth":"A","point":[90,120],"reference":true}],
       "2":[{"truth":"B","point":[160,120],"reference":true}],
       "8":[{"truth":"A","point":[100,120]},{"truth":"B","point":[145,118]}]}}
```

The report counts two correctly retained IDs, missing and duplicate poses, and joins to the partner.
A tie in torso alignment abstains. Include both performers before and after a crossing.

Compare encoder batches on the same labelled VTT selection and GPU using
`python -m benchmarks.benchmark_crowded_vtt --config facial_recognition.yml --labels pairs.json --body-batches 1 2 4`.
Each batch starts with a separate temporary database and retains the configured pose backend.
Compare cue throughput, embedded bodies per second, peak PyTorch CUDA allocation and the labelled assignment results.

### Compare recognition through a crowded VTT contact

In a JSON file, label each visible person's torso in the original VTT cue
pixels. Use the same truth label for that person across cues and scenes.
Choose clean, face-visible reference cues for both performers. A wrist label
is optional and uses the person's anatomical left or right.

```json
{
  "123": {
    "120": [{"truth":"A","point":[90,75],"reference":true},{"truth":"B","point":[160,85],"reference":true}],
    "121": [{"truth":"A","point":[110,78],"wrists":{"right":[152,86]}},{"truth":"B","point":[142,83]}],
    "122": [{"truth":"A","point":[87,82]},{"truth":"B","point":[176,85]}]
  }
}
```

Run the comparison from the repository root with Stash credentials and an
enabled SapiensID collection/identity configuration. The example measures GPU
activity on CUDA. On CPU omit `--cuda-profile`.

```shell
python -m benchmarks.benchmark_crowded_vtt --config facial_recognition.yml --labels contact-labels.json --compare-contact --cuda-profile
```

The variants use RF-DETR-XL with ViTPose++, RF-DETR-XL with masked
PMPose, and two variants with an additional full-frame RTMO-S pass (box or torso-aware NMS) on
unresolved face/pose counts, overlapping boxes or temporal two-person
evidence. RTMO's pinned COCO17 ONNX model downloads into the configured model
directory if missing-model downloads are enabled. All variants scan the
complete selected scenes, check their cue pixels and timings against the
first run, rebuild identities using private databases, then score only the
labelled cues. A reference ID comes from the matched torso’s accepted strict face
anchor. A missing face reference is counted explicitly. No Stash memberships are published. Compare missing/duplicate
poses, partner false joins, both correct, anchored query recovery, retry count,
and CUDA activity seconds per 1000 cues. A missing reference is reported
separately. Run the experiment again on the same file to compare future changes.

The contact comparison also runs RTMO with torso-aware suppression on the
same ONNX outputs. The original RTMO variant uses box NMS. Both check that
each previously associated face still belongs to the same physical torso.
The report includes future-context retry attempts per 1000 scanned cues and
the number of cues that yielded recovered bodies, including bracketed empty
cues. These retries use the neighbouring retained VTT pixels and timestamps.

### Optional paired KPR retrieval

Install the [upstream KPR project](https://github.com/VlSomers/keypoint_promptable_reidentification/tree/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa)
and its dependencies in its own Python 3.10 environment. Pass that interpreter
via --kpr-python. Recognition runs in the normal stash_ai environment. Use the trained Occluded
PoseTrack checkpoint and matching YAML. Pass both local paths explicitly.

```shell
python -m benchmarks.benchmark_crowded_vtt --config facial_recognition.yml --labels contact-labels.json --compare-contact --kpr-python /opt/kpr/bin/python --kpr-config /models/kpr/kpr_occ_posetrack_test.yaml --kpr-weights /models/kpr/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar
```

Mark clean reference cues for at least two performers. The paired retrieval
experiment uses the RF-DETR/ViTPose boxes, identical verified VTT pixels and
the same labelled references for both encoders. KPR takes crop-relative
positive joints and visible negative joints of other detected people.
SapiensID uses its retained body embedding. Each query ranks the enrolled
performers by mean distance to their references, excluding its own cue and
identical crops. The output counts correct, partner, foreign and tied or
unscorable retrievals plus missing inputs. Wrist annotations give a separate
breakdown for poses with a predicted wrist closer to the partner's labelled
wrist. This closed-set ranking measures the encoder signal separately from
the end-to-end identity assignments. The optional model only runs in the
benchmark and its working files use the configured benchmark directory.
