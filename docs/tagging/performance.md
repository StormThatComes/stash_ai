# Performance and memory

[Documentation index](../README.md) · [Tagging workflow](README.md)

- [Default limits](#default-limits)
- [One model on the GPU](#one-model-on-the-gpu)
- [Shared input between models](#shared-input-between-models)
- [Merge cache](#merge-cache)
- [SQLite](#sqlite)
- [Dry-run](#dry-run)
- [Controlling memory](#controlling-memory)
- [Measurement](#measurement)
- [Worker configuration](#worker-configuration)
- [Input and result pipeline](#input-and-result-pipeline)

The current pipeline is optimized for sequential
model inference and a large local merge cache.

## Default limits

| Option | Default | Meaning |
|---|---:|---|
| `--ram-budget-mb` | 32768 | Whole-process memory target in MB |
| `--ram-cache-mb` | 24576 | Shared cache in MB for results, indexes and media |
| `--prepare-workers` | 4 | Parallel CPU frame transformations |
| `--checkpoint-batches` | 8 | Save progress every N processed batches (integer >= 1) |
| `--prefetch-batches` | 4 | Prepared batches ahead of inference |
| `--result-prefetch-batches` | 32 | CPU result batches buffered during post-processing and writes (1–128) |

Cache allocation is lazy and also limited to 75% of
the RAM budget. The budget is **not a hard process cap**:
models, tensors, active scenes, Python and SQLite
also consume memory.

Global `checkpoint_batches` in `tagger_models.yml`
controls inference checkpoint frequency. CLI
`--checkpoint-batches N` overrides YAML. Missing
settings retain default 8. Larger intervals save
less frequently but can require replaying more batches
after a hard process kill. Final checkpoints and writes
on handled interruption/errors are independent of
this interval. Changes do not invalidate results/checkpoints.

## One model on the GPU

Models run sequentially. Image preparation can use
multiple CPU workers, but inference from separate
models does not overlap. This limits peak VRAM
and simplifies recovery.

## Shared input between models

Within a merged run, later models may reuse downloaded
VTT/sprites. The application retains input hashes and
rejects changes to data between model passes.

There is no persistent image cache on disk.

## Merge cache

After publication, `merge_checks` records fingerprints
of policy and scene state. An identical subsequent
run can skip legacy import, evidence decoding, index
building, marker union and publication planning.

Result, snapshot and cleanup writes invalidate the
corresponding checks transactionally.

Changes to publication code or configuration may
require one re-merge but do not necessarily
require re-inference.

## SQLite

Results are compressed. Publication saves a `pending`
plan and promotes it to `published` after successful
Stash synchronization.

Derived records and compact indexes improve performance,
while source evidence remains authoritative for
merge/cleanup replay.

## Dry-run

For current database formats, dry-run attaches the
source read-only and writes differences into a
temporary overlay, avoiding an in-memory copy of
the entire state database.

## Controlling memory

Lower limits if the system starts swapping:

```powershell
python .\stash_ai.py --models all --ram-budget-mb 20480 --ram-cache-mb 14336
```

For unusually large scenes, one active structure
may temporarily exceed the typical cache share.
The final report includes peak cache, sampled RSS,
hits, misses, evictions and pressure trims.

## Measurement

```powershell
python -m benchmarks.benchmark_performance --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
```

Benchmarks are synthetic and include neither GPU
inference nor a real Stash server. Use `--profile-merge`
to diagnose a particular library; see
[merge profiling](merge-profiling.md).
## Worker configuration

### Worker counts in tagger_models.yml

Global settings independently control worker counts for each stage:

```yaml
checkpoint_batches: 8
prefetch_batches: 4
result_prefetch_batches: 32
prepare_workers: 4
merge_workers: 4
model_report_workers: 4
```

`prepare_workers` controls CPU image transformation,
`merge_workers` controls evidence reading/replay, and
`model_report_workers` controls model-report processes.
Worker counts are integers from 1 to 16; `1` means sequential
execution. Corresponding CLI options (`--prepare-workers`,
`--merge-workers`, `--model-report-workers`) override YAML.
Without these keys, defaults remain 4/4/up to 4 available CPUs.

Merge and model reports read worker settings from an existing
YAML file without loading models. Missing YAML does not prevent
replaying saved evidence: defaults or CLI settings apply.
Other offline commands do not read this file for worker counts.
Changes to worker counts do not invalidate inference identity
or persisted results. Model inference and Stash writes remain
sequential.

## Input and result pipeline

- Global `prefetch_batches: 4` in `tagger_models.yml` (0–128; zero
  disables prefetch). Explicit `--prefetch-batches N` takes precedence;
  omission retains the default 4. This controls preparation of future
  batches independently of inference `batch_size` and does not
  invalidate stored predictions.
- Global `result_prefetch_batches: 32` (1–128), or
  `--result-prefetch-batches N`, separately controls the CPU result
  queue. It applies when `prefetch_batches > 0`. CLI overrides YAML,
  and the default remains 32 when omitted. Changing it does not
  invalidate results or checkpoints.

With `prefetch_batches > 0`, one persistent thread loads a model,
runs inference for all its scenes, then closes it. A separate
backend thread is **not** created for every scene, so CUDA resources
and stream pools can retain a single owner. Meanwhile, the main
thread processes previous results and persists checkpoints/SQLite.
The result queue holds at most `result_prefetch_batches` batches;
one more result may be pending insertion and one inference call may
be in flight. Results are consumed in order and checkpoints cover
only fully processed prefixes. Failure in a later batch preserves
earlier progress. Interruption cancels scene preparation and waits
for inference, after which the model is closed on its owning thread.
Prefetched inputs and queued results occupy **RAM, not VRAM**.

A larger result queue allows inference to continue while previous
results undergo slower CPU post-processing, compression and saving.
This queue occupies RAM **outside the cache limit**. Float32 result
arrays alone use approximately
`result_prefetch_batches × batch_size × label_count × 4` bytes,
plus any in-flight result and frame metadata. The queue buffers
temporary stalls; if the consumer is consistently slower than GPU
inference, it will still fill up.

Prefetch also handles input for **one next scene**. Downloading
VTT/sprites, verifying their hashes, decoding images and building
CPU batches can overlap with inference of the current scene.
It uses a separate HTTP session and the shared RAM cache.
The media buffer retains at most two compressed sprites within
16 MiB; the shared cache still respects its memory budget.
The next scene's input queue holds `prefetch_batches` prepared
batches, potentially with one batch waiting to enter it and
a transformation window of up to
`max(batch_size, prepare_workers)` frames.

Both scenes share one `prepare_workers` pool, so the thread count
does not double. Input verification uses its own executor, and at
most two CPU batch producers have isolated HTTP sessions.
A full current-scene queue does not block fetching or preparing
the next scene. Each active input queue has a separate capacity:
current and next scenes can both retain batches simultaneously.
These are live CPU inputs and lie outside `--ram-cache-mb`.
Larger prefetch depth raises memory use proportionally to
`prefetch_batches` and `batch_size`; it does not increase the
number of look-ahead scenes or the GPU result-queue depth.
Scenes already eligible for reuse are not prefetched.

Read failures are surfaced only when their scene is reached,
so they do not interrupt saving previous results. The same
holds for next-scene decoding or inference errors.

After the final batch of the current scene, the main thread may
consume ready next-scene input, validate reuse, and restore its
checkpoint. Only then can it start that scene on the **same GPU
owner**, before finalizing, assembling and committing the previous
scene. Its output waits in the bounded RAM queue; checkpoints and
completed results are still written in scene order on the main
thread. If next-scene input is still downloading or verification
failed, the previous scene finalizes without waiting. Handoff
is attempted again before result assembly and commit if input
became ready in the meantime.

During resume, speculative CPU batches from the beginning of the
next scene are discarded; the GPU processes only the remaining
frames after the verified checkpoint. Ctrl+C also cancels
next-scene download and inference; unprocessed GPU results do
not extend the saved checkpoint.
`--prefetch-batches 0` disables look-ahead scene preparation too.

CPU preparation constructs complete inputs (stacking, masks and
fixed-size ONNX batch padding) before passing them to inference.
Transformation workers can process frames from multiple batches:
a batch size of 4 no longer restricts 8 workers to four simultaneous
tasks. Media/PIL reading continues to have one owner. Besides the
batch queue, the transformation window is limited to
`max(batch_size, prepare_workers)` frames. With prefetch disabled,
inference and result processing stay on the main thread with
the existing transformation pool.

Run history separates `sprite_frame_read`, `image_prepare`,
`batch_collate`, `inference_input_wait`, `inference_batch`,
`inference_result_wait`, `inference_result_queue_wait` and
`inference_postprocess`. A full result queue suggests that the
consumer cannot keep up; a long `inference_input_wait` suggests
input preparation is the bottleneck. `scene_input_prefetch`
measures background verification, while `scene_input_wait`
measures waiting for prepared next-scene input.
`scene_batch_prefetch` and `scene_batch_queue_wait` cover
CPU preparation and waiting for space in the next-scene queue.
`scene_handoff` measures authorizing and starting the next
scene before finalization of the previous one.
`inference_scene_finalize`, `inference_output_capture` and
`inference_commit` distinguish CPU finalization, result assembly
and persistence, to help locate pauses at scene boundaries.
On the synchronous path `input_verify` still includes reading
and hashing; with prefetch it covers receiving input, with
waiting tracked separately by `scene_input_wait`.

`inference_batch` includes transfers and validation, not just
CUDA kernel execution. Parallel stage timings may overlap and
must not be summed as wall-clock runtime. Pipeline changes
require neither database migration nor rejection of compatible
results/checkpoints.

Results and checkpoints use faster zlib level-1 compression.
Older BLOBs remain readable; faster writes may increase database
size. `state_json_encode` and `state_compress` track result
serialization; `checkpoint_json_encode` and
`checkpoint_compress` cover checkpoint preparation, including
top-score array compression. `checkpoint_prepare` includes
BLOB preparation before the transaction, `checkpoint_sqlite_write`
covers SQL and commit, and `checkpoint_write` includes both.

`--db-compact` retains the write lock until final database validation
and post-VACUUM counter comparison.

Parallel merge divides the RAM and cache budget between helpers
and the parent process responsible for publication. With four
helpers, each process gets one-fifth of the budget; once they
exit, the parent regains the entire budget. Model-report workers
use the same eligibility settings and limits as the sequential
report.

Memory CLI limits use MB (1 MB = 1048576 bytes, equivalent to MiB) and accept
fractional values, for example `--ram-cache-mb 512.5`. The defaults are 32768 MB
for the process target and 24576 MB for caches. Replace the former `--ram-budget-gb`
and `--ram-cache-gb` options by their `-mb` equivalents and multiply old GiB values
by 1024. The old options are rejected to avoid ambiguous units.
