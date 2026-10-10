# Performance and memory

The current pipeline is optimized for sequential
model inference and a large local merge cache.

## Default limits

| Option | Default | Meaning |
|---|---:|---|
| `--ram-budget-gb` | 32 | Whole-process memory target |
| `--ram-cache-gb` | 24 | Shared cache for results, indexes and media |
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
python .\stash_ai.py --models all --ram-budget-gb 20 --ram-cache-gb 14
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
[merge profiling](UPGRADE_diagnostics.md).
