# Tagging state and cache reuse

[Documentation index](../README.md) · [Tagging workflow](README.md)

- [State and cache behavior](#state-and-cache-behavior)
- [Rethresholding example](#rethresholding-example)

## State and cache behavior

Before skipping a merge based on cached state, Stash AI refreshes the scene's
current state in Stash. SQL reads are batched; if another automation changed a
marker after the initial scan, a fresh merge is triggered.

Inference identity includes SHA-256 hashes of the model files and JoyTag
normalization (`JOYTAG_CLIP_MEAN` and `JOYTAG_CLIP_STD`).
Weight changes are detected even when file size and modification time are
unchanged. Each model's files are hashed once per run; after downloading
missing assets only the new files are hashed. The first run following this
identity change will refresh older inference results and checkpoints.

The default `tagger_sources.sqlite3` database stores raw model evidence,
checkpoints, published snapshots, and metadata for safe resume.
**Back up this file.** It contains neither images nor the API key.

In `merged` mode, the default resume behavior:

- skips completed inference when the model and input identities match;
- resumes a compatible partial checkpoint;
- can re-run merging without a GPU after publication-policy changes;
- uses retained observations after threshold changes when their coverage
  is sufficient.

`--restart` forces fresh inference for the selected models/scenes. Do not
use it merely to re-run merge, cleanup, or threshold evaluation.

### Evidence retention independent of publication thresholds

Optional `scene_cache_floor` and `marker_cache_floor` settings in each
model's `tagger_models.yml` section retain weaker predictions for later
threshold adjustments without a GPU. They **do not** lower the thresholds
for publishing scene tags or markers:

```yaml
wd:
  scene_threshold: 0.80
  marker_threshold: 0.85
  scene_cache_floor: 0.20
  marker_cache_floor: 0.25
```

Values must be in the range 0–1. Omitting a field or setting it to `null`
retains the previous automatic retention behavior. The effective persistence
floor is the minimum of the configured value and the support threshold
required by the current publication policy. A higher configured floor
therefore does not discard observations needed for publication.
The scene cache also stores exact aggregates for weaker labels.

After fresh inference, lower the publication thresholds in YAML and run
`--merge-only --rethreshold-scenes --rethreshold-markers`, provided the
persisted retention floors cover the newly required support thresholds.
Changing retention itself during a normal run triggers fresh inference and
invalidates a partial checkpoint; `--merge-only` cannot recover predictions
that were previously filtered out. Lower retention floors increase database
size and cache-processing cost. These settings affect persistent inference
evidence, not publication thresholds.

An active cache requires explicit retention floors and complete analysis
format 5. `--db-update` retires older inference; run inference again before
merge or cleanup. Overriding publication thresholds cannot restore
observations that were previously discarded.

The rethreshold `Check` and `Apply` phases read compact
`result_metadata` rows in SQLite instead of decoding frames from
`results.data`. New results write this metadata atomically with evidence;
older results populate it on the first check. Each model and result kind
has one active record and an optional numeric publication threshold.
Replacing or removing raw results invalidates their metadata via a trigger,
including writes by older program versions. `Apply` stores thresholds in
`result_thresholds` and updates `result_index` using metadata: result
presence and the current record's most recent inference signature.
When metadata is available, neither phase decodes frame BLOBs.
Dry run only populates a temporary database; `--db-check` verifies
existing metadata against evidence without repairing it.

New inference always replaces the previous result for the given model and
output kind. Upgrading from v18 retires incomplete results stored in older
analysis formats and schedules reinference. Publications retain ownership
and geometry, with required evidence pinned in the archive. Normal merging
and cleanup require native intervals; `OccurrenceTag` points remain points.
Legacy ordinary AI points, YAML v1 hierarchies, and managed `jt:`, `wd:`,
`dtq:` and `op:` prefixes require the
[transition-release migration tools](../maintenance/legacy-migrations.md#migration-tools-from-the-transition-release).
Current code only publishes the `ai:` namespace and does not convert
legacy per-model objects. Manual associations with old tag definitions
are preserved.

## Rethresholding example

Example: inference with `scene_threshold: 0.80` and
`scene_cache_floor: 0.20` retains observations from 0.20
plus exact label aggregates. Lowering `scene_threshold`
to 0.40 with `scene_support_ratio: 0.85` requires evidence
down to 0.34, so `--rethreshold-scenes` can use that cache
without loading a model. Lowering it to 0.15 would require
0.1275 and is rejected due to insufficient retained evidence.

Actual floors are saved in `scene_score_floor` and
`marker_score_floor` metadata. Changing retention settings
can invalidate completed inference and partial checkpoints
during normal execution. Fresh inference is then needed
to recover previously discarded observations; merging
and rethresholding cannot reconstruct them. Keep floors
unchanged when only lowering publication thresholds.
Broader retention costs storage and may increase merge
RAM/CPU usage. Fresh inference always persists evidence
under the configured retention rules.

### Deferred input verification (schema v26)

Run `python stash_ai.py --db-update` once for existing databases.
This adds a small input-verification table without changing raw inference
records or publication snapshots. Missing VTT/sprite paths defer
publication and retain previous predictions. After restoring identical
inputs, `--resume` verifies and reuses compatible results instead of
forcing inference. Model-utility reports exclude scenes awaiting input
verification or fresh inference and include skipped counts.

Explicit missing scene IDs are reported separately. During resumed
`--db-init`, a successful Stash read confirming scene deletion retires
that scene so remaining scenes can complete. Network/read errors
cannot authorize retirement.

For a detailed cache-based publication preview, use
`--merge-only --dry-run --dry-run-plan review.jsonl`.
Event details and symbolic IDs are described in
[the dry-run plan](publication-preview.md).

### Rerunning without refetching verified inputs

By default, normal execution/resume trusts already verified input
for each completed model/scene. It skips fetching VTT and sprites
from Stash and skips model loading, even if another selected model
still needs to process that scene. It continues to check model
identity, model-file contents, vocabulary, result completion and
retained evidence's threshold coverage. Compatible changes to
batch size, workers or publication thresholds do not force inference.

After regenerating VTT/sprites **at the same URLs**, force a fresh
input verification:

```powershell
python .\stash_ai.py --force-fetch-vtt
```

This option fetches and verifies VTT and relevant sprites even
for completed scenes. It does **not** force inference:
unchanged inputs with compatible results are reused. Only
results with modified inputs/models or insufficient evidence
coverage are recomputed. This applies to normal inference/resume,
not reports, `--merge-only` or standalone rethreshold commands.

New/incomplete results and checkpoints still fetch their inputs
during the corresponding model pass. For example, if JoyTag
finished but Oppai was interrupted, JoyTag skips fetching
while Oppai fetches input and resumes. If that later model
detects a changed VTT/sprite, previously saved results based
on the old inputs are immediately blocked and queued for
refresh before the shared merge. The block persists through
interruption or failure. Results saved with different inputs,
pending verification, changed tracked paths, or sampling limits
also require verification.

No database migration or inference/checkpoint format change is
needed. Existing v27 identity records work immediately with
the default full sampling. Older entries did not record paths;
run `--force-fetch-vtt` after changing those paths.
Missing identity or legacy identity with reduced sampling
requires input verification, after which compatible results
may still be preserved. New/verified records store a small
checksum and input context without rewriting raw results.
Run history marks skipped fetches using `input_fetch_skipped`;
it neither pretends that verification occurred nor authorizes reuse.

### Republishing dependencies

The merge fingerprint covers modules responsible for replaying
retained evidence and publishing tags/markers, along with the
effective policy, publication options, exclusions, hierarchy
and observed catalog/scene. Changes limited to backend, checkpoints,
input preparation, prefetch, worker settings or history do not by
themselves invalidate a completed publication because a source
file changed. New results and actual scene/policy changes still do.

The dependency manifest is conservative at the module level:
changing a combined storage or domain module may still require
rechecking merge. A one-time manifest change rechecks existing
merge checks against stored results, without altering inference
identity or result/checkpoint/snapshot formats.

Pinned published/pending/recovery evidence is lazily read once
during a scene+marker result write. The read is not reused
across subsequent writes, rollbacks or publication changes.
The complete archive of the specific pinned proofs is retained.

See [database maintenance](../maintenance/database.md) for schema upgrades and repairs,
and [legacy migrations](../maintenance/legacy-migrations.md) for retired conversion tools.
