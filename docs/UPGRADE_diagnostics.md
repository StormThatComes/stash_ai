# Merge profiling

Use `--profile-merge` to investigate slow merges or merges
that unexpectedly recompute scenes. Combine it with
`--merge-only` to avoid new inference.

## Collecting a profile

```powershell
python .\stash_ai.py --merge-only --scene-id 123 --force-merge --profile-merge merge-profile.jsonl
```

For multiple scenes:

```powershell
python .\stash_ai.py --merge-only `
  --scene-id 123 --scene-id 124 --scene-id 125 `
  --force-merge --profile-merge merge-profile.jsonl
```

The `.jsonl` file must be new; the application refuses
to overwrite an existing profile. `--force-merge` bypasses
only publication cache; it neither deletes evidence nor
runs models. `--verbose` adds terminal detail but is not
required for profiling.

## JSONL contents

The profile contains a start record, per-scene records
and a summary. Available scene fields include:

| Field | Meaning |
|---|---|
| `wall_seconds` / `cpu_seconds` | Elapsed time and CPU time |
| `status` / `reason` | Outcome and reason for recomputing/skipping |
| `stages` | Stage durations and call counts |
| `api` | GraphQL/SQL operations, durations, errors and sizes |
| `counts` | Input/output counts, cache hits and bytes |
| `statistics` | Planned publication changes |
| `unmeasured_seconds` | Time outside recorded stages |

Stages have exclusive and inclusive timing. Use exclusive
time when estimating a stage's contribution, not the sum
of a parent and its children.

## Primary areas

Typical stage groups include marker/scene/Occurrence
computations; local state SQL, decode/encode, compression
and cache; Stash reads and response decoding; tag
resolution/catalog work; and preparing, mutating and
committing publication.

The initial collection scan is reported globally; it
is not part of an individual scene's final merge time.

## Diagnosing a bottleneck

If an identical later run still recomputes a scene,
start with `reason` and the `merge_*` counters.
Common reasons include changed publication code/policy,
changed Stash tags or markers, new saved inference results,
pending publication, or changes to the managed tag catalog.

If API time dominates, inspect operation counts and whether
batched read-only SQL is active. If local state dominates,
compare decoding, encoding, compression and snapshot sizes.
If compute dominates, profile fusion/interval-union
algorithms next.

## Profile confidentiality

Profiles do not store API keys, images or full GraphQL
queries and parameters. They do contain scene IDs, stage
names and quantitative data needed for analysis.

## Offline benchmark

For a reproducible synthetic comparison:

```powershell
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
```

The benchmark does not contact production Stash and
is not a substitute for profiling the actual library.
