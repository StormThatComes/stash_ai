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

The `rfdetr-vitpose-rtmo-missing` profile only retries missing people with two face or
temporal torso cues. `delta_from_vitpose` reports the change in correct pairs,
partner joins, missing/duplicate poses and time per 1,000 cues.

The optional paired KPR report also removes the query's true identity from the
reference gallery. Each model calibrates its distance cutoff at the same target
false-accept rate (1%) using absent probes from other labelled scenes. It reports
actual held-out false accepts, correctly recovered known queries and abstentions.
Include at least two scenes with competing references. A single scene reports
missing calibration. These retrieval results do not change production assignments.

### Fragmentation at library scale

Run the retained assignment pipeline for increasing scene counts with:

```sh
python -m benchmarks.benchmark_fragment_pipeline --config facial_recognition.yml --scenes 32 128 1000 10000
```

Each scene defaults to 300 VTT cues with two detected people and two face
records in seven of every ten cues. The fixture varies sampling intervals,
view, outfit and recurring actors. Rear face detections have no usable face
embedding. Both full and incremental runs use the configured work directory,
disk and memory budgets. At 10,000 scenes body vectors alone occupy about
64 GiB. The benchmark releases each temporary database before the next run.

Each JSON line reports local planning and global reconciliation time, sampled
coordinator RSS, SQLite reads, available process I/O counters, solver trials
and the physical assignment partition. Compare the incremental result before
and after revalidating the complete selection against the cold partition.
A small smoke run can use `--scenes 2 --cues 10 --batch 1 --solver-trials 2`.

## Run pipeline profiling

Set `execution.rebuild.profile: true` in the active v2 YAML and run recognition
normally. The existing `runs.summary.pipeline_profile` field now uses format 2.
It records scalar counters and timings only, with no scene URLs, SQL text,
embeddings, images or per-cue events. The number of fields does not grow with the
number of scenes. Profiling does not change model settings or force CUDA
synchronization.

- `stages` contains elapsed time on the coordinator thread.
- `worker_stages` contains summed elapsed time in explicitly instrumented media,
  pixel preparation, PMPose preparation and scan-writer tasks. These durations
  overlap both the coordinator and other workers; they are not CPU utilization.
- `counts` combines bounded counters from these participants, including HTTP
  requests/downloaded bytes, sprite cache lookups and actual sprite decodes.
- Every timer includes `calls`, `errors` and `seconds`. Normal iterator exhaustion
  is counted as a call but no longer as an error. Actual source failures still
  propagate and increment the error counter.

Useful stage families:

| Stages | What they include |
| --- | --- |
| `scene_input_prepare`, `media_read`, `media_http_headers`, `media_http_body` | Source verification and HTTP work, including scene-prefetch workers |
| `frame_materialize`, `sprite_decode`, `frame_input_wait`, `frame_queue_put` | Synchronous body-frame reads, sprite decoding, consumer waits and producer backpressure |
| `face_model_load`, `person_detector_model_acquire`, `pose_model_acquire`, `body_model_acquire`, `sam_model_load`, `model_release` | Lazy initialization/acquisition and phase cleanup; acquisition can include restoring parked CPU weights |
| `face_detection_batch`, `face_detection`, `face_embedding_batch`, `face_embedding` | Face inference API calls, including their cache/adapter overhead |
| `body_enrichment`, `person_detection`, `pose_estimation`, `body_embedding` | Sequential pose/backfill work and the detector/pose/identity stages |
| `pmpose_first_pass`, `pmpose_mask_refinement`, `pmpose_masked_second_pass` | Initial pose estimation and the mask-assisted refinement pipeline |
| `sam_image_embedding`, `sam_mask_prediction`, `mask_prompt_core`, `mask_validation` | SAM image features, per-person prompting, prompt selection and mask/box checks |
| `pmpose_prepare`, `pmpose_prepare_wait`, `pmpose_inference` | CPU crops, their consumption and inference/decoding, across both pose passes |
| `pixel_prepare`, `pixel_result_wait`, `face_crop_encode`, `body_prepare_and_encode`, `body_crop_encode`, `body_prepare_wait` | Pixel-worker work, encoding and waiting for prepared body inputs |
| `scan_sqlite_write`, `scan_writer_wait`, `body_sqlite_checkpoint`, `body_sqlite_write` | Complete-cue scan transactions and body-evidence persistence; the body-write timer alone excludes the caller's commit |
| `publication`, `auto_prune`, `retention_plan`, `retention_delete` | Publication and retention after recognition |

Timers are inclusive. For example, `pmpose_masked_second_pass` is inside
`pmpose_mask_refinement`, and `sam_image_embedding` is also inside refinement.
Do not add nested stages or worker/coordinator durations to estimate total time.
GPU-facing timers are host elapsed time, not CUDA kernel time; asynchronous work
can finish at a later transfer or existing synchronization boundary. Acquisition
timers may be nested in the first inference call.

The recognition outcome is first saved durably with `cutoff: before_auto_prune`.
After retention succeeds, fails or is interrupted, the same summary's pipeline
profile is refreshed with `cutoff: after_auto_prune_before_profile_write` and
`retention_outcome`. This final snapshot includes retention and model cleanup,
but excludes the final telemetry write and outer database/cache teardown. The
existing run status and `finished` timestamp retain their recognition-completion
meaning. An interrupted process or failed telemetry write can leave the earlier,
explicitly labelled snapshot; it does not invalidate the recognition outcome.

Export the latest finalized summary after the command exits, for example:

```powershell
sqlite3 -readonly faces_state.sqlite3 "SELECT summary FROM runs WHERE finished IS NOT NULL ORDER BY started DESC LIMIT 1;" > run-profile.json
```

Include the active YAML and code revision with the report. Compare fresh and
reused scene/cue counts before comparing timings. A cache lookup miss is not an
operational error, and a batch maximum does not describe actual batch occupancy.
