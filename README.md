# Stash AI

Automatic scene tagging and time-based marker generation for [Stash](https://github.com/stashapp/stash).

Stash AI supports four models: JoyTag (`jt`), WD EVA02-Large v3 (`wd`),
DanbooruTagQuery L/16 (`dtq`), and OppaiOracle V1.1 (`op`). Models run
sequentially, with only one model loaded on the GPU at a time.

The default `merged` mode publishes shared `ai:` tags, combines compatible
model predictions, and retains per-model provenance and confidence in a local
SQLite database. Visual input comes exclusively from Stash VTT files and
sprites; the application never opens video files.

## Requirements

- **Python 3.14.7** — the only supported and tested version;
- **Stash 0.31.1 or newer**, with generated sprites/VTT;
- Dependencies from `requirements.txt`;
- CUDA is recommended. CPU inference works but is substantially slower.

The Stash version is checked before reading scenes or changing server data,
including during `--dry-run`. Local commands such as `--db-update`,
`--db-check`, and `--db-report` do not require a Stash connection.
Known intervals always publish an end time; occurrences and imported markers
whose end is unknown have `end_seconds: null`.

For CPU-only environments, replace `onnxruntime-gpu` with a compatible
`onnxruntime>=1.21,<2` version.

## Quick start

```powershell
pip install -r requirements.txt
# Pinned Python 3.14.7 environments: see constraints/README.md

$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<your Stash API key>"

python .\stash_ai.py --download-models --models all
python .\stash_ai.py --models all --scene-id 123 --dry-run --verbose
python .\stash_ai.py --models all --scene-id 123
```

Set `STASH_API_KEY` explicitly in your environment if your Stash instance
requires authentication. Without this variable, the client does not send an
`ApiKey` header. A historical embedded key has been removed; see
[SECURITY.md](SECURITY.md) for key-rotation guidance.

In `tagger_models.yml`, every selected model must explicitly define
`scene_threshold`, `marker_threshold`, and `batch_size`. Models are selected
using `--models` or the then-required `enabled_models` setting; missing
values do not fall back to legacy JoyTag defaults. Optional
`scene_cache_floor` and `marker_cache_floor` control evidence retention.
Resume reuses matching cached results, but the mere presence of published
tags or markers does not bypass inference.

## Common operations

```powershell
# Normal inference; matching cache entries and checkpoints are reused automatically.
python .\stash_ai.py --models all

# Force fresh inference for a selected scope.
python .\stash_ai.py --models wd --scene-id 123 --restart

# Re-merge stored results without running any models.
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Parallel read-only evidence preparation; Stash publication remains sequential.
python .\stash_ai.py --merge-only --merge-workers 4
# --dry-run and --profile-merge intentionally use the sequential path.

# Apply new thresholds to retained cache without running models.
python .\stash_ai.py --models all --merge-only --rethreshold-markers
python .\stash_ai.py --models all --merge-only --rethreshold-scenes --rethreshold-markers

# Clean up excluded results.
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only

# Repair groups of existing markers.
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run

# Inspect the locally recorded evidence for a tag.
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123

# Remove unused tag definitions.
python .\stash_ai.py --prune-empty-tags-only --dry-run

# Inspect, migrate, validate, and compact the local state database.
python .\stash_ai.py --db-report
python .\stash_ai.py --db-check
python .\stash_ai.py --db-update --dry-run
python .\stash_ai.py --db-update
python .\stash_ai.py --db-compact --dry-run
python .\stash_ai.py --db-compact

# Create a local anonymous diagnostic report for optional manual sharing.
python .\stash_ai.py --anonymous-diagnostics .\stash-ai-diagnostics.json

# Model utility and leave-one-model-out ablation without inference.
# Up to four CPU processes by default; --model-report-workers 1 forces sequential execution.
python .\stash_ai.py --model-report .\model-utility.json
python .\stash_ai.py --model-report .\model-utility.json --model-report-workers 8
python .\stash_ai.py --model-report .\model-utility-scene-123.json --scene-id 123

# Tags worth reviewing: reasons, examples, and per-model counts.
python .\stash_ai.py --tag-review-report .\tag-review.json
python .\stash_ai.py --tag-review-report .\tag-review-scene-123.json --scene-id 123

# Cache completeness against YAML and a plan to backfill missing scene/model pairs.
python .\stash_ai.py --cache-report .\cache-coverage.json
python .\stash_ai.py --cache-report .\cache-coverage-all.json --models all
python .\stash_ai.py --cache-report .\cache-coverage-scene-123.json --models wd --scene-id 123

# Abandon an unfinished merge/cleanup plan.
python .\stash_ai.py --abandon-pending
```

Threshold changes write small rows to the `result_thresholds` table;
compressed inference results and frame-level evidence remain unchanged.
This also applies when changed YAML thresholds trigger automatic cache reuse.
Reading results, merging, cleanup, and tag explanation use the effective
thresholds from that table. Each model/output pair has one active result and
an optional numeric `result_thresholds.threshold`. Databases predating v18
must first use the upgrade bridge described below. Supported schemas use
the single-record/numeric-threshold representation.
`--dry-run` stores thresholds only in a temporary database.

Merge snapshots store shared `extra_support` evidence once and reference it
from contributions, reducing serialization and compression costs.
The current database schema is **v27**, with upgrades supported from **v18**.
New snapshots use format envelope **v7**. `--db-update` converts snapshot
formats v4/v5/v6/v7, including pending and recovery data, in one pass,
preserving interval geometry and publication ownership, normalizing frozen
replay settings, and removing `extra_ids` and redundant
`algorithm: temporal` fields. Full pre-format-5 inference payloads are
retired; dependent publications receive `requires_rebase` and need fresh
inference followed by `--merge-only`. The format-5 evidence archive remains
available for exact reconstruction of published results. Checkpoint-analysis
versioning affects incomplete inference only: obsolete checkpoints are
ignored, and inference restarts that model/scene from the beginning.
Completed format-5 results retain evidence and signatures; older active
results must be recomputed. Dry runs change only a temporary database.

`--explain-tag` outputs scene reports with `format: 2`. Each item in
`recorded_sources` has one `record` object containing `meta`,
`thresholds` and `detections`, rather than the former one-item
`records` array.

To see every option:

```powershell
python .\stash_ai.py --help
```

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

### Starting a new database with existing AI results

`--db-init` can rebuild state after the database is lost, or start a fresh
database while retaining existing Stash results until each scene finishes
inference:

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3
```

A new or empty database and both output types (scene tags and markers) are
required. The command adopts existing `ai:<label>` tag assignments and
application-owned markers. After inference succeeds for all selected models
it removes outdated AI results. Missing VTT/sprites or model errors leave
the scene's previous results intact. Manually assigned `ai:` tags are
also adopted, except category and reserved marker tags.

Resume an interrupted run with the same command **without** `--db-init`,
keeping the model selection, database path, and any `--scene-id` scope.
Repeating the flag also preserves progress. Dry run uses only a temporary
overlay and writes neither to Stash nor to the source database.
See [database initialization](docs/UPGRADE_db_init.md).

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
[transition-release migration tools](#migration-tools-from-the-transition-release).
Current code only publishes the `ai:` namespace and does not convert
legacy per-model objects. Manual associations with old tag definitions
are preserved.

### Migration tools from the transition release

Three completed migrations are no longer included in the current package.
If needed, run them from a separate checkout of immutable commit
[`9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
The current package uses schema v27. Databases older than v18 first need
the earlier upgrade bridge described in the database maintenance section.
Back up the database and configuration, and install the checkout's
dependencies in a separate Python 3.14 environment.

```powershell
git clone https://github.com/StormThatComes/stash_ai stash_ai_migration_bridge
git -C stash_ai_migration_bridge checkout --detach 9fcaa52502cd487455457547860b5db7d8710ad8
cd stash_ai_migration_bridge

# Convert custom v1 hierarchies; specify all hierarchy files in use.
python -m stash_ai.commands.hierarchy_migration C:/stash_ai/joytag_hierarchy.yml C:/stash_ai/wd_hierarchy.yml C:/stash_ai/dtq_hierarchy.yml C:/stash_ai/oppai_hierarchy.yml --dry-run

# Legacy AI points: derive end_seconds from published/pending/recovery state.
python -m stash_ai.tools.point_migration --state-db C:/stash_ai/tagger_sources.sqlite3 --dry-run

# Legacy prefixes: rerun inference and publish in the shared ai: namespace.
python -m stash_ai.tools.namespace_migration --models all --state-db C:/stash_ai/tagger_sources.sqlite3 --model-config C:/stash_ai/tagger_models.yml --temporal-policy C:/stash_ai/temporal_policy.yml --marker-core-exclusions C:/stash_ai/marker_core_exclusions.yml --scene-tag-exclusions C:/stash_ai/scene_tag_exclusions.yml --hierarchy C:/stash_ai/joytag_hierarchy.yml --wd-hierarchy C:/stash_ai/wd_hierarchy.yml --dtq-hierarchy C:/stash_ai/dtq_hierarchy.yml --oppai-hierarchy C:/stash_ai/oppai_hierarchy.yml --dry-run
```

Replace `C:/stash_ai` with the absolute path of your existing installation.
Use the same `STASH_URL` and `STASH_API_KEY`, and preserve model selection,
policies, and output scope. If hierarchies are disabled, replace the four
hierarchy options with `--no-hierarchy`. Once the plan is verified, repeat
the relevant commands without `--dry-run`. YAML conversion saves originals
as `*.v1.yml` and retains group assignments and source metadata.
Point migration does not load models, and retains IDs, titles, and manual
tags; occurrences remain points. Prefix migration preserves manually
managed objects and marker extras; matching markers keep their IDs.
Its scope can be limited with `--scene-id`, `--markers-only`, or
`--scene-tags-only`. Finish an interrupted prefix publication with the same
migrator and configuration before returning to the current version.
Pending snapshots remain available for API failure recovery.
Then return to the current installation and perform `--db-check` plus
any required inference.

### Database health, upgrades, and compaction

`--db-update` upgrades the database selected by `--state-db` from
schema **v18 or newer** to the current **v27**, across all scopes.
Normal execution and `--dry-run` require the current schema; new databases
start on v27. Migration adds database-rebuild state but does not adopt
existing Stash results. Adoption is authorized only by an explicit
`--db-init` on a new/empty database.

The updater uses the regular writer lock and runs offline, without Stash or
model configuration. It retains numeric thresholds on current results,
publication ownership, pending publication/cleanup and recovery state.
Older results are marked for reinference and their checkpoints removed.
Snapshot versions v4/v5/v6/v7 are converted in one pass, preserving shared
evidence. Retired algorithm settings are removed while geometry and evidence
references stay intact. Updating cannot reconstruct lost detections.
Pre-format-5 inference requires a rerun.

Result metadata is populated after conversion. Frames in active results
must be chronologically ordered. For databases predating v23, unordered
records are marked for reinference and dependent publications receive
`requires_rebase`. Stored payloads are not sorted or re-signed; ordered
format-5 results remain valid. Reads and writes reject missing provenance,
threshold and retention-floor metadata rather than silently dropping votes.

Normal cleanup accepts only the current fingerprint. An interrupted cleanup
with a legacy fingerprint requires an explicit offline conversion using
**the same YAML files and output scope** (`--cleanup-only`,
`--cleanup-markers-only`, or `--cleanup-scene-tags-only`). The tool validates
every per-model policy, updates only the pending key, and retains the
generation, masks and recovery state:

```bash
python -m stash_ai.tools.cleanup_pending_migration --state-db /path/to/tagger_sources.sqlite3 --cleanup-only --marker-core-exclusions marker_core_exclusions.yml --scene-tag-exclusions scene_tag_exclusions.yml --dry-run
```

Remove `--dry-run` after checking the plan, then resume the original cleanup.
The migration tool neither contacts Stash nor loads models.

For databases **older than v18**, first use compatible commit
[`c5bcd887bafe76b7ee2a9de944666799c9a0fc30`](https://github.com/StormThatComes/stash_ai/commit/c5bcd887bafe76b7ee2a9de944666799c9a0fc30)
in a separate directory with its dependencies. Back up the database and
point the bridge at the same state file:

```bash
git clone https://github.com/StormThatComes/stash_ai stash-ai-upgrade-bridge
git -C stash-ai-upgrade-bridge checkout c5bcd887bafe76b7ee2a9de944666799c9a0fc30
python stash-ai-upgrade-bridge/stash_ai.py --db-update --state-db /path/to/tagger_sources.sqlite3
```

Then run `--db-update` from the current installation. The compatibility
commit converts historical series/imports to v20, which the current
release can upgrade directly.

`--db-update --dry-run` reports source and target schema versions without
migration, file writes or locking. Databases older than v18 receive an
upgrade-path message; foreign or newer schemas are rejected. Payload
conversion, retiring active results, and updating the version execute
within one transaction. Missing-table creation is idempotent, and an
interrupted upgrade can be retried.

`--db-check` performs a deep, read-only consistency check across every
database scope. On older schemas, it only checks the SQLite integrity
and schema preflight, and requires `--db-update` before deep validation.
In addition to a full SQLite `integrity_check`, it validates:
- `results` against `result_index`, including recorded last-inference signatures;
- decoding of persistent BLOBs and valid model/output kinds;
- checkpoint/frame/analysis relationships and checkpoint dimensions;
- cleanup/recovery generations;
- required scene, frame and marker-interval fields, tags, times,
  probabilities and saved temporal statistics;
- checkpoint frames, including `scene_tags`, and float32 values
  (rejecting `NaN`, infinities and values outside `[0, 1]`).

Active results must satisfy the current inference contract. Legacy
intervals and incomplete statistics are only accepted in archived evidence.
Imports and `score: null` assessments are rejected. Retired
`legacy_imports` and `refinement_progress` tables are flagged without
reading their payloads. Some missing snapshot/recovery associations are
warnings. Pending publication/cleanup and refresh requests are treated as
in-progress work, not automatically as errors.

The check does not connect to Stash, load models, repair data, or modify
anything. A progress bar displays the current phase (e.g. SQLite integrity,
results/index, state payloads, checkpoints). Exit code `0` indicates no
logical errors, `1` indicates inconsistencies; warnings alone do not fail it.

`--db-report` needs neither Stash nor models. It reports schema version,
SQLite `quick_check`, DB/WAL/SHM size, free pages, result counts grouped by
model/output kind, and checkpoint, refresh, pending/recovery and merge-check
counts for the current scope. It does not print tag names, scene IDs or scores.

`--db-compact` runs SQLite `VACUUM` under the normal writer lock. It first
checks database integrity, free disk space, and the absence of unfinished
merge/cleanup operations. `--dry-run` runs only the preflight and estimates
space reclaimable from unused pages. Recovery evidence and checkpoints
are preserved.

### Anonymous developer diagnostics

`--anonymous-diagnostics NEW.json` creates a local report for manual
inspection and, optionally, sharing with a developer. The application
**never uploads it automatically** and refuses to overwrite an existing file.

The report uses `format: 2`. For v18–v20 databases, it reports basic
database information and `database.upgrade_required: true` without decoding
evidence or computing detailed statistics. Generate a full report after
`--db-update`. Legacy modes, provenance and analysis formats are identified
as `other`, without dedicated fields for historical intervals or
`scene_frames_format`.

The report covers the entire selected state database but exports no scene
IDs, `scope` values, URLs, file paths, tag names, scores, marker timestamps,
signatures or input hashes. Unrecognized model/output names become `other`;
custom tables are reported only by count.

For schema optimization, the report includes:
- Python/SQLite versions and operating-system family;
- page size/count, freelist, DB/WAL size, number of scopes and scenes;
- record counts for known tables and BLOB-size distributions;
- per model/output kind: compressed JSON size percentiles, compression
  ratio, items per record, detections per frame and scene observations;
- aggregate checkpoint, result-index and refresh-request state.

Although it removes direct identifiers and content, exact counts and size
aggregates can still characterize a particular installation. Review the
generated JSON before sharing it.

### Run history and profiling report

Inference, merge, cleanup and state-writing commands automatically record
run history in a separate `<state-db>.runs.sqlite3` file. Set
`--run-history-db PATH` to choose another file or `--no-run-history` to
disable recording. Run history is independent of cache and does not alter
the cache schema, signatures, or resume rules. Back it up together with
state if you need historical comparisons.

Inspection and reporting commands do not record history by default. Use
`--record-run` to profile, for example, `--model-report`, `--cache-report`
or `--db-check`. `--dry-run` and programmatic `DRY_RUN` never write history,
even with `--record-run`. History recording does not force sequential
merges or model reports: CPU workers return their aggregates to the parent.

```powershell
python .\stash_ai.py --models all
python .\stash_ai.py --merge-only
python .\stash_ai.py --model-report .\models.json --record-run
python .\stash_ai.py --run-report .\runs.json
python .\stash_ai.py --run-report .\wd-runs.json --run-model wd --run-command inference --run-status finished --run-limit 100
```

`--run-report NEW.json` reads only the history database, prints a summary,
and creates a new JSON file. It does not need the cache database, models,
or Stash. Filters cover command, status and model; a limit of 1–10000
selects the newest matching runs (50 by default). Existing report files
are not overwritten.

Each run records status, wall time, parent CPU time, configuration hashes,
worker settings, models and revisions, available weight/vocabulary hashes,
change and error counts, and stage/API timing. API keys, media URLs,
payloads, and error contents are not recorded. This is **not** an
anonymous report: run IDs, dates, configuration and aggregates describe
your installation. Run start is saved before execution; Ctrl+C and errors
finalize the record. Partial aggregates are flushed at stage boundaries,
every 30 seconds, or after 32 completed model/scene/merge work units.
`last_progress_at` and `progress_seq` identify the most recent update;
`partial=true` means a run did not finish and is not counted as a
successful complete run. `running` means no end was recorded: the process
may still be running or may have been killed. History-write failures
neither stop the real command nor mask its errors.

The report includes available timings for scenes, inference batches,
frame preparation, checkpoints, cache, rethreshold, merge, publication
and API calls. Fresh inference, resume and cache reuse are distinguished.
Metrics include call counts, total/mean/min/max durations, approximate
p50/p95 from bounded histograms, and throughput where work-unit counts
are available. Mean is total time divided by calls, including when
aggregating runs. Comparisons group compatible configurations, environments,
and model identities.

Configuration includes a hash of the sorted set of selected scene IDs
(order and duplicates do not change the hash), code hash, available Git
revision, and installed library versions. `execution_observed=false`
means the model's execution device was not observed, for example when
cache was reused. Verified inputs have a separate workload identity:
pair counts, cue counts and input fingerprints, used for grouping.
These diagnostics do not authorize cache reuse or change stored prediction
identities. The report separates completion statuses and successful/failed
operations.

Stage timings include inclusive time and self time. Threads/processes
may overlap, so summed stage times are not elapsed run time.
`cpu_seconds` covers the parent; `worker_cpu` covers measured worker tasks.
A failed worker task has `outcome=error`, increments `errors` and
does not count as a completed scene (`units=0`), even when delivered
to the parent as a result. Merge continues with other scenes after
such errors; the model report indicates failure.

Histograms have bounded size; history grows by one record per run without
storing every timing sample. Filtered/limited reports summarize only
selected runs, not necessarily all of history.

### Dlaczego scena została przeliczona?

Schemat v27 dodaje zweryfikowaną tożsamość inferencji w bazie stanu. Uruchom
`python .\stash_ai.py --db-update` przed użyciem starej bazy. Migracja zachowuje
oryginalne BLOB-y wyników, checkpointy i frozen proofs; nie uruchamia modeli.
Przy pierwszym zgodnym odczycie CLI zapisuje brakującą tożsamość bez inferencji.
Następne zmiany `batch_size`, lokalizacji identycznych wag lub mniej wymagającej
retencji korzystają z tych samych scores, jeżeli weryfikacja wejść i pokrycia
cache przejdzie. Zmiana batcha może dawać drobne różnice numeryczne przy świeżym
obliczeniu; reuse zachowuje dotychczasowe wyniki, nie obiecuje bitowej zgodności.

Dla starego cache bez zarejestrowanej tożsamości wykonaj zgodny przebieg z
dotychczasową konfiguracją przed przenoszeniem wag lub zmianą batcha/retencji.
Jeżeli starej tożsamości nie można udowodnić, CLI nie zgaduje jej na podstawie
historii diagnostycznej. Zmiana wag/wejść i brak wymaganych scores nadal mogą
uzasadnić przeliczenie konkretnej pary scena/model. Zgodne checkpointy można
wznawiać po zmianie batcha; ich polityka przechowywania musi pozostać zgodna.

Zwykła inferencja zapisuje w historii decyzję dla każdej pary scena/model:
`infer`, `cache_reused`, `rethreshold`, `checkpoint_resumed` albo `blocked`.
Dodaj `--verbose`, aby widzieć decyzje i powody podczas pracy. Osobny raport
czyta te same dane offline, drukuje podsumowanie per model i zapisuje nowy JSON:

```powershell
python .\stash_ai.py --reuse-report .\reuse.json
python .\stash_ai.py --reuse-report .\wd-scene.json --models wd --scene-id 123
python .\stash_ai.py --reuse-report .\recent-reuse.json --run-limit 20 --run-status finished
```

Domyślnie raport wybiera ostatni pasujący przebieg z zapisanymi decyzjami.
`--run-limit` wybiera 1..10000 takich przebiegów; filtry modeli (`--models`
lub `--run-model`), scen (`--scene-id` można powtarzać), komendy i statusu
stosowane są przed limitem. Późniejszy merge lub maintenance bez decyzji
inferencji nie ukryje ostatniego raportu. Raport opisuje wykonane przebiegi;
nie sprawdza bieżących plików ani dostępności mediów w Stash.

Powody obejmują brak evidence, jawny restart, wyłączone resume, oczekujący
refresh, niekompletny indeks oraz zmianę modelu, rewizji, wag, słownika,
batch size, agregacji, retencji, progów, VTT lub sprite. Jeśli dostępna jest
poprzednia tożsamość, JSON zawiera wartości przed/po, liczebność cue i medianę
odstępów między ich początkami. Dla zbyt płytkiego cache pokazuje zapisany
floor i wymaganą retencję; dla checkpointu liczbę wznowionych klatek i powód
przyjęcia lub odrzucenia. Kilka powodów może dotyczyć jednej decyzji, więc
ich liczebności nie muszą sumować się do liczby scen.

Starszy cache bez historii nadal działa normalnie. Po zgodnym reuse można
uzupełnić jego tożsamość diagnostyczną bez inferencji. Gdy wcześniejszych
danych brakuje, `historical_identity_unavailable` oraz ogólne `input_changed`
lub `model_or_configuration_changed` wskazują granice dostępnej informacji.
Diagnostyka nie zmienia sygnatur, schematu cache ani warunków jego użycia.
Historia w formacie 1 jest odczytywana bez modyfikacji i aktualizowana do
formatu 2 dopiero podczas zapisu kolejnego przebiegu.

Historia przechowuje identyfikatory scen i modeli, powody, parametry oraz
hashe wejść, bez adresów mediów i kluczy API. Decyzje rosną wraz z liczbą
par scena/model/przebieg; poprzednia tożsamość zajmuje jeden rekord na
scenę/model/rodzaj wyniku. Zapis jest buforowany w krótkich transakcjach po
maksymalnie 64 operacje, a normalne zakończenie i Ctrl+C opróżniają bufor.
Nagłe zabicie procesu może utracić ostatnie niezapisane operacje historii.
`--no-run-history` wyłącza ich zapis; raport wymaga wcześniej zebranych danych.
Bez `--verbose` pomija również budowanie diagnostycznych proofs. Weryfikacja
wejść i tożsamości cache nadal działa. Błędy metadanych GPU i opcjonalnej
historii nie zmieniają wyniku polecenia ani decyzji o ponownej inferencji.

### Postęp inferencji

W terminalu inferencja pokazuje dwa paski: sceny bieżącego modelu oraz klatki
aktualnej sceny, np. `wd scene 6306: 128/729 frame`. Pasek klatek liczy cue
z VTT wybrane do inferencji, a nie wszystkie klatki wideo. Przy wznowieniu
zaczyna od poprawnie odtworzonego checkpointu i przesuwa się po obróbce
każdego batcha przez CPU, także gdy model nie zachował żadnych tagów.
Prefetch i inferencja następnej sceny nie przesuwają paska bieżącej sceny.
Po osiągnięciu 100% trwa jeszcze finalizacja i zapis wyniku; pasek scen
przesuwa się dopiero po zakończeniu obsługi sceny. Sceny użyte z cache
nie uruchamiają paska klatek. Pasek klatek jest ukryty poza interaktywnym
terminalem, żeby przekierowane logi nie zawierały dwóch linii z ruchem kursora.

Pod paskami pojawiają się dwa wiersze z zajętością buforów, np.:

```text
Buffers: in=12/16 out=5/32 | next[6307]: in=16/16 out=0/32
Cache: 1.2GiB/24.0GiB | checkpoint interval=7/32 batch
```

`in` to gotowe batche wejściowe względem `prefetch_batches`, a `out` to
wyniki GPU w RAM czekające na obróbkę CPU względem `result_prefetch_batches`.
`next[ID]` pokazuje osobne kolejki następnej sceny, także przed przekazaniem jej
na GPU. Licznik nie obejmuje batcha już pobranego przez konsumenta ani zadań
preprocessingu, które jeszcze nie skończyły się przygotowaniem batcha. Sygnały
końca i błędy nie są liczone jako batche. Przy `prefetch_batches: 0` kolejki
pokazują `off`; brak następnej sceny również pokazuje `next=off`.

`Cache` to szacowane bajty zatrzymane we wspólnym cache RAM i jego rzeczywisty
skonfigurowany limit, po ograniczeniu przez `ram_budget_gb`. To wspólny cache
mediów, wejść, wyników i indeksów, a nie całe zużycie RAM procesu lub VRAM.
`checkpoint interval` pokazuje liczbę przetworzonych batchy od ostatniego
udanego zapisu względem `checkpoint_batches`; nie jest pojemnością kolejki.
Po zapisie checkpointu licznik wraca do zera.

Status jest odczytywany przy odświeżeniu paska klatek (domyślnie nie częściej
niż co 0,25 s) oraz na granicach finalizacji i zapisu sceny. Są to chwilowe
odczyty pracujących równolegle buforów, bez czekania na producerów, pobierania
elementów, dostępu do SQLite lub zmian LRU. Dodatkowe wiersze są ukryte poza
interaktywnym terminalem i zamykane także po błędzie lub Ctrl+C.

### Model Utility / Ablation Report

`--model-report NEW.json` działa wyłącznie na zapisanym evidence i nie łączy się
ze Stash ani nie uruchamia modeli. Dla każdej sceny najpierw odtwarza normalny
wynik wszystkich zapisanych modeli, a następnie cztery kontrfaktyczne wyniki
`all minus one model`. Oznacza to, że po usunięciu modelu od początku
przeliczane są consensus seeds, temporal rules, shared scene tags, marker extras
i Occurrence — raport nie filtruje jedynie gotowych markerów.

Per model raport pokazuje m.in.:
- liczbę scen z zapisanym evidence i scen, w których pominięcie modelu zmienia wynik;
- finalne scene tags i marker components wspierane przez model;
- scene tags, markery, extras i Occurrence utracone po jego usunięciu;
- markery, które pozostają, ale zmieniają zakres lub extras;
- procent scen z evidence, w których model ma mierzalny marginalny wpływ.

Zmiana samego badge'a źródeł markera, np. `[jt+wd]` na `[wd]`, nie jest liczona
jako zmiana semantyczna. Dzięki temu raport mierzy wpływ na tagowanie, zakresy i
extras, a nie samą obecność provenance. `--scene-id` może ograniczyć analizę
do wybranych scen.

### Raport tagów wymagających przeglądu

`--tag-review-report NEW.json` analizuje zapisane tagi scen i główne etykiety
markerów, odtwarzając merge z efektywnymi progami zapisanymi w bazie. Używa
wybranej polityki temporalnej i wykluczeń. Nie ładuje modeli, nie pobiera mediów,
nie łączy się ze Stash i nie zmienia bazy źródłowej ani publikacji.

Sceny z `input_checks` (`missing_media`, `read_error`, `deleted`) są pomijane
z powodem `unverified_inputs`. Dotyczy to również problemu innego modelu w tej
samej scenie; raport zachowuje zapisane wyniki i podaje przyczynę w przykładach.

Raport wskazuje priorytety ręcznego przeglądu:

- `weak_or_borderline`: najlepszy wynik nie przekracza właściwego progu o więcej
  niż `--tag-review-margin` (domyślnie `0.05`). Dla scen zakwalifikowanych przez
  `strong_peak` używany jest próg silnego piku; pozostałe wyniki mają własny próg
  sceny lub markera. Słabe wyniki zaakceptowane przez reguły czasowe/konsensus też
  mogą zostać wskazane;
- `isolated_peak`: tag sceny kwalifikuje się przez silny pik, ale tylko jedna
  próbka przekracza próg sceny, przy co najmniej dwóch próbkach;
- `model_disagreement`: inny model zachował tę samą etykietę na identycznym
  wejściu, lecz nie kwalifikuje jej według własnych reguł sceny. Brak modelu lub
  etykiety w cache nie jest głosem negatywnym; wykluczone etykiety nie służą do
  porównań;
- `long_marker`: po scaleniu zakres ma co najmniej
  `--tag-review-long-marker-seconds` (domyślnie `300` sekund).

Tabela w terminalu i sekcja `models` JSON pokazują dla JoyTag, WD, DTQ i Oppai
liczbę różnych etykiet do przeglądu, liczbę par scena–tag oraz liczbę scen.
Para scena–tag per model jest liczona raz, również gdy ten tag ma kilka zakresów,
występuje jako tag sceny i core markera lub pochodzi z kilku zapisanych przebiegów.
Globalne liczniki dodatkowo deduplikują modele; sumowanie wierszy modeli nie daje
liczby różnych tagów w całej kolekcji. Liczniki powodów mogą się nakładać.
Model wspierający core markera głosem konsensusowym również jest uwzględniony,
nawet jeśli nie utworzył własnego zakresu. Jego wynik pochodzi ze zgodnych,
zachowanych obserwacji w zakresie wkładu, a przykład ma `support_only: true`.

Sekcja `tags` zawiera wiersze model–etykieta, liczniki powodów i do trzech
przykładowych scen z wynikami, progami i czasami. Limit przykładów zmienia
`--tag-review-examples 1..20`; liczniki obejmują wszystkie przeanalizowane sceny.
Wiersze są sortowane według liczby wskazanych scen, następnie liczby powodów.
`--scene-id` może ograniczyć analizę; istniejący plik raportu nie jest nadpisywany.

Sceny z pending refresh/publication lub nieużywalnym evidence są pomijane
i raportowane osobno z powodami, zamiast uczestniczyć w porównaniu modeli.
Raport opisuje wynik replay cache, a nie sprawdzony na żywo stan Stash.
Heurystyki nie oznaczają potwierdzonych błędów: poprawny tag może mieć tylko jeden
pik, długi zakres albo wsparcie jednego modelu. Raport nie uśrednia ocen modeli.

### Raport kompletności cache i plan uzupełnienia

`--cache-report NEW.json` sprawdza oba rodzaje evidence (sceny i markery) dla
modeli włączonych w YAML lub wybranych przez `--models`. Działa offline, bez
modeli, słowników, mediów i połączenia ze Stash. Nie zmienia źródłowej bazy
ani publikacji i nie nadpisuje istniejącego raportu.

Domyślny zakres obejmuje sceny znane **bieżącemu scope lokalnej bazy**, również
te obecne tylko w checkpointach, refresh requests, snapshotach, recovery lub input_checks.
Sceny ze Stash, które nigdy nie trafiły do tej bazy, nie wchodzą do mianownika.
Powtarzalne `--scene-id` ustala dokładny zakres; ID bez żadnego lokalnego stanu
jest wtedy raportowane jako brakujące. Raport nie sprawdza aktualności wag,
słownika, VTT ani sprite'ów — ich zgodność potwierdza zwykły przebieg inferencji.

Tabela terminala i `models` w JSON pokazują per model liczbę par scena–model:
gotowych, możliwych do przeliczenia bez GPU, wymagających inferencji,
wymagających wyłącznie rethreshold oraz zablokowanych. `outputs` rozbija
liczniki osobno na scene/markers i rozłączne statusy:

| Status | Znaczenie |
|---|---|
| `ready` | Ukończona inferencja, zgodny indeks, efektywne progi i wystarczająca retencja |
| `rethreshold` | Cache obejmuje progi YAML; wystarczy przeliczenie bez GPU |
| `missing` | Brak wyniku danego rodzaju |
| `partial` | Brak wyniku, lecz istnieje checkpoint do weryfikacji przy wznowieniu |
| `refresh_required` | Oczekujące żądanie świeżej inferencji unieważnia stary wynik |
| `insufficient` | Brak evidence klatek, za wysoki floor lub wcześniej odfiltrowane odblokowane etykiety |
| `inconsistent` | Wynik i indeks są niespójne albo evidence jest nieczytelne; sprawdź `--db-check` |

Ukończony wynik bez żadnych detekcji jest poprawnym cache, a nie brakiem.
`evidence_ready_pairs` opisuje pokrycie zapisanych wyników. Znane odroczone
kontrole wejścia blokują `fully_ready_scenes` i cały plan dla danej sceny,
również gdy dotyczą niewybranego modelu. `issues.input_checks` podaje model
i przyczynę. Przywróć dostępność mediów lub wyjaśnij usuniętą scenę, następnie
zweryfikuj wejście zwykłym uruchomieniem; sama blokada nie wymaga `--restart`.
Retencja jest sprawdzana względem minimum floor z YAML i progu wsparcia
potrzebnego do publikacji. Sprawdzany jest jeden aktywny wynik każdego modelu i wyjścia.
Przy gotowych `result_metadata` raport nie dekoduje BLOB-ów predykcji; starsze
metadane uzupełnia wyłącznie w tymczasowej bazie. Głęboka kontrola payloadów
pozostaje zadaniem `--db-check`.

`issues` zawiera konkretne pary scena–model z powodami, wymaganymi progami,
blokadami oraz zapisanym postępem checkpointu. `plan` zawiera uporządkowane,
pogrupowane **tablice argumentów `argv`**, po maksymalnie 50 scen na polecenie:

- `resume_or_infer` uzupełnia tylko wskazane modele/sceny; zwykłe resume sprawdza
  zgodność checkpointu i wykorzystuje poprawne ukończone wyniki;
- `restart_inference` wymusza wymianę nieużywalnego ukończonego cache. Aktywne
  braki/refresh z checkpointem dostają zwykłe resume, aby nie kasować postępu;
- `rethreshold` stosuje oba progi z YAML bez inferencji;
- `merge_after_backfill` na końcu odświeża konsensus modeli i Occurrence.

Polecenia zachowują ścieżki `--state-db`, YAML modeli, polityki temporalnej
i wykluczeń. Raport **nie wykonuje planu**. Ręczne uruchomienie tych poleceń
może zmienić publikację w Stash; używaj tego samego `STASH_URL` i ustawień
środowiska. Przy wielu oczekujących refreshach publikacja może pozostać
zablokowana do ukończenia wszystkich wskazanych modeli.

Pending publication, globalny cleanup, refresh niewybranego modelu,
nieczytelne evidence lub wyłączone zastępowanie nieużywalnej historii blokują
wszystkie polecenia planu dla dotkniętej sceny, również uzupełnienie innych
modeli i końcowy merge. Każde z tych poleceń może próbować publikacji.
Najpierw dokończ pierwotny plan
albo wyjaśnij niespójność, a następnie wygeneruj raport ponownie. Gotowy evidence
może nadal mieć blokadę publikacji; liczniki tych dwóch cech mogą się nakładać.

Merge używa także zachowanego evidence niewybranych modeli. Raport sprawdza
jego możliwość publikacji według zapisanych efektywnych progów i wybranej
polityki, bez wymagania retencji z YAML dla tych modeli. Niepełny lub nieczytelny
cache takiego modelu blokuje scenę; `issues[].publication_dependencies` wskazuje
model, rodzaj wyniku, powód i sposób rozwiązania. Dołącz wskazany model przez
`--models` i wygeneruj raport ponownie, aby zaplanować jego naprawę. Liczniki
kompletności i cele inferencji nadal dotyczą wybranych modeli; raport nie
rozszerza automatycznie zakresu inferencji. `issues[].blocking_models` wskazuje
wybrane modele, których blokada uniemożliwia zaplanowanie pozostałych par sceny.

Exit code `0` oznacza poprawnie wygenerowany raport, również gdy zawiera braki.
Archiwalne evidence po `--abandon-pending` nie blokuje cache ani inferencji:
`publication_recovery_scene_ids` pokazuje te sceny, a końcowy merge planu
rekonsyliuje ich publikację, również gdy cache jest już kompletny, jeśli scena
nie ma innych blokad publikacji.

## Wynik w Stash

W trybie `merged`:

| Element | Przykład |
|---|---|
| Tag sceny / core markera | `ai:blue_hair` |
| Tytuł markera | `[AI] blue_hair [jt+wd+op]` |
| Tag własności | `ai:marker:stash-taggers` |
| Punkt Occurrence | dodatkowo `ai:marker:occurrence` |
| Grupa zwykłego markera | `m:blue_hair` |
| Grupa Occurrence | `m:blue_hair:occurance` |
| Korzeń grup markerów | `ai_markers` |
| Kategorie hierarchii | `ai:category:...` |

Pisownia `:occurance` w nazwach grup jest celowa i pozostaje zgodna z istniejącym
formatem.

Nowe wyniki zawsze używają wspólnej publikacji `ai:` i reguł temporalnych.
Przełączniki `--output-mode` i `--no-temporal` oraz klucz YAML `enabled`
zostały usunięte; usuń je ze skryptów i `temporal_policy.yml`. Poszczególne
reguły nadal konfiguruje polityka YAML, w tym `occurrence_markers`.
Historyczne namespace’y modeli przenosi osobny migrator; zwykłe publikowanie,
cleanup i domyślne pruning obsługują bieżące wyniki `ai:` oraz zarządzane grupy
`m:`. Definicje dawnych prefiksów można usuwać jawnie, np.
`--prune-empty-tags-only --prune-prefixes jt: wd: dtq: op:`. Ręczne dodatki
z innych namespace’ów pozostają zachowane.

Nowa inferencja zapisuje kompletny format analizy 5: sygnatury wejścia i
inferencji, jawne progi retencji oraz statystyki lub jawne `null` dla
niekwalifikujących się agregatów. `--db-update` wycofuje starsze aktywne wyniki,
które wymagają ponownej inferencji. Przypięte archiwum zachowuje dokładne dowody
publikacji, a geometria i ownership dawnych snapshotów pozostają dostępne dla
recovery i migracji punktów. Cleanup wymaga przebudowy wycofanych publikacji
przez świeżą inferencję i `--merge-only`.

Nowy fingerprint cleanupu zawsze zawiera zakres wyjść i nie zawiera stałego
pustego blocklist. Przerwany cleanup zachowuje generację i maski. Historyczny
podpis wymaga jawnego narzędzia `stash_ai.tools.cleanup_pending_migration`,
które sprawdza ten sam zakres wyjść i politykę wykluczeń. Nowe operacje zapisują tylko bieżący podpis.

## Konfiguracja

Wszystkie modele korzystają z jednego pliku `tag_hierarchy.yml`, będącego unią
dotychczasowych czterech hierarchii: 21 154 unikalne tagi w 92 grupach.
Opcja `--hierarchy YAML` wskazuje wspólny plik również przy wyborze jednego modelu.
Usunięto `--wd-hierarchy`, `--dtq-hierarchy` i `--oppai-hierarchy`.

Hierarchia ma `version: 2` i pola `source` oraz `groups`. Ścieżki grup tworzą
bezpośrednio kategorie `ai:category:` pod korzeniem `ai:category:AI`.
Walidacja wymaga przypisania wszystkich tagów wybranych modeli; tagi pozostałych
modeli są dozwolone. Opcjonalne `source.tag_count` i `source.vocabulary_sha256`
opisują cały zbiór tagów wspólnego pliku, niezależnie od wybranych modeli.
Zmiana grupowania nie wymaga ponownej inferencji ani nowej bazy SQLite;
przypisania można zsynchronizować przez `--sync-hierarchy-only`.
Przy własnych zmianach w starych plikach przenieś je do wspólnego YAML-a;
aktualizator zachowuje stare konfiguracje lokalne, ale bieżący kod ich nie czyta.
Własne YAML-e v1 przekonwertuj za pomocą [wersji pomostowej](#migratory-w-wersji-pomostowej).
Bieżący walidator wymaga v2 i wskazuje commit migratora bez zmiany pliku.

| Plik / opcja | Rola |
|---|---|
| `tagger_models.yml` | modele, katalogi, progi, batch size |
| `temporal_policy.yml` | reguły temporalne, consensus, extras i Occurrence |
| `scene_tag_exclusions.yml` | globalne wykluczenia tagów |
| `marker_core_exclusions.yml` | wykluczenia głównych tagów markerów |
| `tag_hierarchy.yml` | wspólna hierarchia kategorii wszystkich modeli |
| `--state-db` | lokalizacja trwałej bazy stanu |
| `--db-init` | odbudowa nowej/pustej bazy z przejęciem istniejących wyników AI |

Publikacja nie ogranicza liczby tagów ani markerów na scenę. Wszystkie wyniki
spełniające progi, reguły temporalne i wykluczenia mogą zostać opublikowane;
dotyczy to każdego modelu, wspólnych wyników i punktów Occurrence.
`MAX_TAGS` i `MAX_MARKERS_PER_SCENE` zostały usunięte. Limit tagów na klatkę
`MARKER_MAX_TAGS_PER_FRAME` pozostaje osobnym ustawieniem (domyślnie 0).

Bazy v18–v26 wymagają `--db-update`. Migracja do v27 zapisuje snapshoty v7
bez dawnych limitów liczby wyników, zachowując geometrię, własność i plany
pending/recovery. Nie zmienia cache inferencji i nie zapisuje do Stash.
Aby uzupełnić wcześniej obcięte wyniki z zachowanego cache, wykonaj:

```powershell
python .\stash_ai.py --db-update
python .\stash_ai.py --merge-only
```

`--merge-only` stosuje aktualne reguły publikacji bez GPU; wymagania dotyczące
kompletności cache nadal obowiązują. `--dry-run` pozwala obejrzeć plan publikacji.

Aktualne domyślne ustawienia pamięci i równoległości:
- `--ram-budget-gb 32`;
- `--ram-cache-gb 24` — cache jest przydzielany na żądanie i ograniczony do
  maksymalnie 75% budżetu;
- `--prepare-workers 4`;
- `--merge-workers 4` — równoległy CPU replay evidence; zapisy Stash/SQLite pozostają sekwencyjne;
- Globalne `checkpoint_batches: 8` w `tagger_models.yml` lub
  `--checkpoint-batches N` wybiera zapis postępu co N przetworzonych batchy.
  Wartość musi być liczbą całkowitą >= 1; CLI ma pierwszeństwo, a brak pola
  zachowuje default 8. Końcowy checkpoint oraz zapis przy obsługiwanym
  przerwaniu/błędzie pozostają niezależne od tego interwału.
- Globalne `prefetch_batches: 4` w `tagger_models.yml` (0–128; 0 wyłącza).
  Jawne `--prefetch-batches N` nadpisuje YAML; brak pola zachowuje default 4.
  Parametr steruje przygotowaniem kolejnych batchy, niezależnie od rozmiaru
  inferencji `batch_size`, i nie unieważnia zapisanych wyników.
- Globalne `result_prefetch_batches: 32` (1–128) lub `--result-prefetch-batches N`
  wybiera osobno pojemność kolejki wyników CPU. Działa przy `prefetch_batches > 0`;
  CLI nadpisuje YAML, a brak pola zachowuje default 32. Zmiana nie unieważnia
  wyników ani checkpointów.

Przy `prefetch_batches > 0` jeden stały wątek ładuje model, wykonuje inferencję
dla wszystkich jego scen i zamyka model. Nie powstaje nowy wątek backendu dla
każdej sceny; zasoby CUDA i pule strumieni mogą zachować jednego właściciela.
Główny wątek równolegle obrabia poprzednie wyniki i zapisuje checkpointy/SQLite. Kolejka
wyników mieści `result_prefetch_batches` batchy; dodatkowo jeden wynik może
czekać na wstawienie lub może trwać jedno wywołanie inferencji.
Wyniki są odbierane w kolejności, a checkpoint obejmuje tylko w pełni
przetworzony prefiks. Błąd późniejszego batcha zachowuje wcześniejszy postęp;
przerwanie anuluje przygotowanie sceny i czeka na inferencję. Następnie model
jest zamykany w swoim wątku, a wykonawca kończy pracę. Wejścia prefetch i wyniki
w kolejce znajdują się w RAM, a nie w VRAM.
Większa kolejka wyników pozwala kontynuować inferencję podczas wolniejszej
obróbki, kompresji i zapisu poprzedniej sceny. Zajmuje RAM poza limitem cache;
same macierze wyników float32 zajmują około
`result_prefetch_batches × batch_size × liczba_etykiet × 4` bajtów,
do czego dochodzi wynik w toku i metadane klatek. To bufor chwilowych opóźnień:
jeśli odbiorca stale pracuje wolniej od inferencji, kolejka nadal się zapełni.

Prefetch obejmuje też wejście jednej następnej sceny: pobranie VTT/sprite'ów
i sprawdzenie ich hashy, dekodowanie obrazów i składanie batchy CPU mogą trwać
podczas inferencji bieżącej sceny. Odczyt używa osobnej sesji HTTP i wspólnego
cache RAM. Lokalny bufor mediów zachowuje najwyżej dwa
skompresowane sprite'y w limicie 16 MiB, a wspólny cache nadal podlega budżetowi
RAM. Kolejka wejść następnej sceny mieści `prefetch_batches` gotowych
batchy; dodatkowo jeden batch może czekać na wstawienie, a okno transformacji
obejmuje najwyżej `max(batch_size, prepare_workers)` klatek. Obie sceny
współdzielą jedną pulę `prepare_workers`, więc jej liczba wątków nie podwaja się.
Weryfikacja wejść ma własnego wykonawcę, a najwyżej dwaj producenci batchy
CPU mają izolowane sesje HTTP. Pełna kolejka bieżącej sceny nie blokuje
pobierania ani przygotowania następnej.
Limit obowiązuje osobno dla każdej aktywnej kolejki wejść: bieżąca i następna
scena mogą jednocześnie zachowywać przygotowane batche. Są one aktywnymi
wejściami CPU, poza limitem cache `--ram-cache-gb`. Większa głębokość zwiększa
zużycie RAM proporcjonalnie do `prefetch_batches` i rozmiaru `batch_size`;
nie zwiększa liczby scen pobieranych z wyprzedzeniem ani kolejki wyników GPU.
Gotowe sceny z planu reuse nie są pobierane z wyprzedzeniem.
Błąd odczytu jest obsługiwany dopiero po dojściu do jego sceny, dzięki czemu
nie przerywa zapisu poprzedniego wyniku. Tak samo obsługiwane są błędy
dekodowania i inferencji następnej sceny.

Po przetworzeniu ostatniego batcha bieżącej sceny główny wątek może odebrać
gotowe wejście następnej, sprawdzić reuse i odtworzyć jej checkpoint. Dopiero
po tych sprawdzeniach uruchamia jej zadanie na tym samym właścicielu GPU,
przed finalizacją, budowaniem wyniku i commitem poprzedniej sceny. Wyniki
następnej sceny czekają w ograniczonej kolejce RAM; checkpointy i pełne wyniki
są zapisywane w kolejności scen, w głównym wątku. Jeśli wejście nadal się
pobiera lub jego weryfikacja zawiodła, finalizacja poprzedniej sceny przebiega
bez czekania na nie. Handoff jest ponawiany także przed budowaniem wyniku
i przed commitem, jeśli wejście stało się gotowe podczas finalizacji.
Przy resume spekulacyjne batche CPU od początku sceny
są odrzucane, a GPU otrzymuje tylko pozostałe klatki po zweryfikowanym checkpoincie.
Ctrl+C anuluje także odczyt i zadanie następnej sceny; nieprzetworzone wyniki
GPU nie poszerzają zapisanego checkpointu.
`--prefetch-batches 0` wyłącza również przygotowanie wejścia następnej sceny.

Przygotowanie CPU składa kompletne wejścia (`stack`, maski i dopełnianie
stałego batcha ONNX) przed przekazaniem do inferencji. Workery transformacji
mogą pracować na klatkach kolejnych batchy: batch 4 nie ogranicza już puli
8 workerów do czterech aktywnych zadań. Odczyt mediów/PIL nadal ma jednego
właściciela. Oprócz kolejki batchy okno transformacji jest ograniczone do
`max(batch_size, prepare_workers)` klatek. `--prefetch-batches 0` zachowuje
inferencję i obróbkę wyników w głównym wątku oraz dotychczasową pulę transformacji.

Historia uruchomień rozdziela `sprite_frame_read`, `image_prepare`,
`batch_collate`, `inference_input_wait`, `inference_batch`,
`inference_result_wait`, `inference_result_queue_wait` i
`inference_postprocess`. Pełna kolejka wyników wskazuje, że odbiorca nie nadąża;
długi `inference_input_wait` wskazuje oczekiwanie na przygotowanie wejścia.
`scene_input_prefetch` mierzy weryfikację następnej sceny w tle,
`scene_input_wait` — oczekiwanie na jej gotowe wejście, a
`scene_batch_prefetch` i `scene_batch_queue_wait` — przygotowanie CPU i
oczekiwanie na miejsce w kolejce następnej sceny. `scene_handoff` mierzy
autoryzację i rozpoczęcie jej zadania przed finalizacją poprzedniej.
`inference_scene_finalize`, `inference_output_capture` i `inference_commit`
rozdzielają finalizację CPU, budowanie wyniku i jego zapis. Pozwala to wskazać
przyczynę pauz na granicach scen. `input_verify` na ścieżce synchronicznej
nadal obejmuje odczyt i hashowanie; przy prefetch obejmuje odbiór wejścia,
z oczekiwaniem wykazanym osobno w `scene_input_wait`.
`inference_batch` nadal obejmuje transfery i walidację, więc nie jest pomiarem
samych kerneli CUDA. Czasy równoległych etapów mogą się nakładać i nie należy
sumować ich jako czasu rzeczywistego przebiegu. Zmiana pipeline'u nie wymaga
migracji bazy ani odrzucenia zgodnych wyników/checkpointów.

Wyniki i checkpointy korzystają z szybkiej kompresji zlib na poziomie 1;
odczyt dotychczasowych BLOB-ów pozostaje zgodny. Kosztem szybszego zapisu
może być większa baza. `state_json_encode` i `state_compress` mierzą zapis
wyników, a `checkpoint_json_encode` i `checkpoint_compress` — przygotowanie
checkpointów, w tym kompresję macierzy top scores. `checkpoint_prepare`
obejmuje przygotowanie BLOB-ów przed transakcją, `checkpoint_sqlite_write`
obejmuje SQL i commit, a `checkpoint_write` obejmuje oba etapy.

`--db-compact` utrzymuje blokadę zapisu aż do końcowej kontroli bazy i
porównania liczników po VACUUM.

Równoległy merge dzieli budżet RAM i cache pomiędzy procesy pomocnicze oraz
proces nadrzędny zapisujący publikacje. Przy czterech procesach pomocniczych
każdy proces dostaje jedną piątą budżetu; po ich zamknięciu proces nadrzędny
odzyskuje pełny budżet. Procesy raportu modeli używają tych samych ustawień
kwalifikacji i limitów co raport sekwencyjny.

## Aktualizacje

`update_stash_ai.py` aktualizuje zarządzane pliki aplikacji bez Gita i bez
uruchamiania `pip`. Wymaga tokena GitHub z dostępem `Contents: read` do
prywatnego repozytorium.

Bieżący updater obsługuje układ pakietowy (`stash_ai/`). Stare instalacje z
modułami `stash_*.py` wymagają jednorazowego przejścia przez updater z commita
[`9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
Zatrzymaj aplikację i pobierz tę wersję jako osobny plik w katalogu instalacji:

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"
$bridge = "9fcaa52502cd487455457547860b5db7d8710ad8"
$headers = @{ Authorization = "Bearer $env:GH_TOKEN"; Accept = "application/vnd.github.raw+json" }
Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/StormThatComes/stash_ai/contents/update_stash_ai.py?ref=$bridge" -OutFile .\update_stash_ai_bridge.py
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout --dry-run
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout
```

Jeśli dawna aktualizacja została przerwana, najpierw uruchom
`python .\update_stash_ai_bridge.py --install-dir . --recover`.
Updater pomostowy instaluje bieżący pakiet, zachowuje backupy, lokalne YAML,
bazę i modele; zmodyfikowany kod wymaga `--force-code`. Następnie używaj
zwykłego `update_stash_ai.py`. Bieżący updater nie skanuje ani nie usuwa
nieśledzonych dawnych modułów. Przy `git pull` do układu pakietowego nie ma
etapu pomostowego updatera.

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"

python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run
python .\update_stash_ai.py
```

Updater:
- pobiera pliki z jednego SHA `main`;
- instaluje launchery, pakiet `stash_ai/` oraz YAML-e; usuwa zastąpione moduły z katalogu głównego;
- zachowuje lokalnie zmienione YAML-e i tworzy propozycje `.incoming`;
- odmawia nadpisania zmienionego kodu bez `--force-code`;
- nie aktualizuje modeli, `.env`, SQLite ani zależności Pythona;
- używa blokady instalacji i dziennika odzyskiwania.

Po przerwaniu aktualizacji użyj:

```powershell
python .\update_stash_ai.py --recover
```

## Bezpieczeństwo

- używaj własnego, zrotowanego `STASH_API_KEY` z environment;
- nie commituj tokenów GitHub ani kluczy Stash;
- read-only GraphQL i media GET mogą być ponawiane po błędach przejściowych;
- mutacje GraphQL **nie są automatycznie retry'owane**, ponieważ serwer mógł
  wykonać zapis mimo utraconej odpowiedzi;
- `--dry-run` nie zapisuje do Stash ani trwałej bazy źródłowej.

## Testy i CI

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

CI testuje Python 3.14.7. Pełny zestaw testów działa na Linux, a updater jest
dodatkowo sprawdzany na Windows. Testy używają atrap Stash i modeli; nie wymagają
produkcyjnego serwera.

## Dokumentacja

- [Aktualny podręcznik](docs/README_multimodel.md)
- [Indeks dokumentacji](docs/README.md)
- [Cleanup SQL](docs/UPGRADE_cleanup_sql.md)
- [Wydajność i pamięć](docs/UPGRADE_performance.md)
- [Merge i szybka ścieżka](docs/UPGRADE_merge.md)
- [Wykluczenia i cleanup](docs/UPGRADE_exclusions.md)
- [Occurrence](docs/UPGRADE_occurrences.md)
- [Grupy markerów](docs/UPGRADE_marker_groups.md)
- [Porzucanie przerwanych operacji](docs/UPGRADE_abandon_pending.md)
- [Odbudowa nowej bazy](docs/UPGRADE_db_init.md)
- [Profilowanie merge](docs/UPGRADE_diagnostics.md)
- [Benchmarki](benchmarks/README.md)

Pliki w `docs/validation/` są historycznymi wynikami konkretnych walidacji i
benchmarków. Nie opisują bieżącej liczby testów ani aktualnego kontraktu aplikacji.

Nazwy konfiguracji i CLI są opisane w [instrukcji aktualizacji wykluczeń](docs/UPGRADE_exclusions.md#aktualizacja-starych-nazw). Używaj `excluded_scene_tags`, `--models`, `--scene-tag-exclusions` i `--cleanup-markers-only`; domyślny plik markerów to `marker_core_exclusions.yml`.


Aktywny wynik to jeden rekord w formacie analizy 5, z opcjonalnym liczbowym progiem
publikacji. Nowa inferencja zastępuje wynik i resetuje próg. Archiwum zachowuje tylko
konkretne dowody przypięte przez published/pending/recovery; zwykły merge, rethreshold,
raporty i indeks ukończenia nie odczytują historycznych segmentów.
`--db-update` zachowuje geometrię, ownership i recovery, a niepełne starsze wyniki
kieruje do ponownej inferencji. `--merge-only` wymaga nowych dowodów przed cleanup.
Brak `end_seconds` w recovery oznacza dokładny punkt i nie pasuje do zakresu.

### Deferred input verification (schema v26)

Run `python stash_ai.py --db-update` once for existing databases. This adds a small
input-verification table without changing raw inference records or publication
snapshots. Missing VTT/sprite paths defer publication and retain previous predictions.
After restoring the same inputs, `--resume` verifies and reuses matching results;
it does not force inference. Model utility reports exclude scenes awaiting input
verification or fresh inference and report the skipped counts.

Explicit missing scene IDs are reported independently. During resumed `--db-init`,
a successful Stash read confirming deletion retires that scene and lets remaining
scenes finish. Network/read errors do not authorize retirement.

Dokładny plan publikacji z cache: `--merge-only --dry-run --dry-run-plan review.jsonl`.
Opis zdarzeń i symbolicznych ID: [docs/dry_run_plan.md](docs/dry_run_plan.md).

### Retencja historii uruchomień

`--prune-run-history` działa wyłącznie na osobnej bazie historii, w scope bieżącego
STASH_URL. Podaj `--history-keep-last N` i/lub `--history-max-age-days DAYS`.
Połączone limity usuwają zakończony przebieg przekraczający dowolny z nich;
`N=0` usuwa wszystkie zakończone przebiegi w tym scope. Przed zastosowaniem możesz
sprawdzić liczbę przebiegów/decyzji i przykładowe ID przez `--dry-run`.

```powershell
python .\stash_ai.py --prune-run-history --history-keep-last 100 --history-max-age-days 90 --dry-run
python .\stash_ai.py --prune-run-history --history-keep-last 100 --history-max-age-days 90
```

Usuwanie przebiegów i ich decyzji jest jedną transakcją. Rekordy `running`,
rekordy bez końca, inne scope i `reuse_proofs` pozostają. Nie są otwierane wyniki,
checkpointy ani snapshoty bazy źródłowej. Historia v1/v2 jest nadal czytelna
bez modyfikacji; pierwszy zapis aktualizuje tylko historię do v3. Retencja nie
uruchamia się automatycznie i nie wykonuje VACUUM.


### Zależności ponownej publikacji

Fingerprint merge obejmuje moduły odtwarzające zachowane dowody i publikujące
tagi/markery oraz efektywną politykę, ustawienia publikacji, exclusions, hierarchy
i zaobserwowany katalog/scenę. Zmiana backendu, checkpointów, przygotowania wejść,
prefetchu, workerów lub historii nie unieważnia gotowej publikacji samą zmianą
pliku. Zapis nowych wyników i rzeczywiste zmiany sceny/polityki nadal ją unieważniają.

Manifest jest konserwatywny na poziomie modułów: zmiana mieszanego modułu storage
lub domain może nadal wymagać sprawdzenia merge. Jednorazowa zmiana manifestu
powoduje ponowne sprawdzenie istniejących merge checks, z użyciem zapisanych wyników;
nie zmienia tożsamości inferencji ani formatów wyników/checkpointów/snapshotów.

Przypięte dowody published/pending/recovery są odczytywane leniwie raz podczas
jednego zapisu wyników scene+markers. Odczyt nie przechodzi na kolejny zapis,
rollback ani zmianę publikacji; pełne archiwum konkretnych dowodów pozostaje.

### Liczba workerów w tagger_models.yml

Globalne ustawienia wybierają liczbę workerów osobno dla każdego etapu:

```yaml
checkpoint_batches: 8
prefetch_batches: 4
result_prefetch_batches: 32
prepare_workers: 4
merge_workers: 4
model_report_workers: 4
```

`prepare_workers` steruje transformacjami obrazów CPU, `merge_workers` odczytem
i scalaniem dowodów, a `model_report_workers` procesami raportu modeli.
Każdy worker count musi być liczbą całkowitą 1..16; `1` oznacza pracę szeregową.
Odpowiedni argument CLI (`--prepare-workers`, `--merge-workers`,
`--model-report-workers`) ma pierwszeństwo przed YAML. Gdy klucza brakuje,
pozostają dotychczasowe domyślne wartości: 4/4/do 4 dostępnych CPU.

Merge i raport modeli odczytują ustawienia workerów z istniejącego YAML, bez
ładowania modeli. Brak pliku nie blokuje pracy na zapisanych dowodach; wtedy
obowiązują wartości domyślne lub CLI. Pozostałe polecenia offline nie odczytują
tego pliku dla ustawień workerów. Zmiana liczby workerów nie zmienia tożsamości
inferencji ani zapisanych wyników. Inferencja modeli i zapisy do Stash pozostają
szeregowe.

### Rerun bez pobierania gotowych wejść

Domyślne uruchomienie/wznowienie ufa wcześniej zweryfikowanym wejściom każdego
ukończonego modelu sceny. Jego przejście nie pobiera VTT ani sprite'ów ze Stash
i nie ładuje modelu, także gdy inne wybrane modele dopiero przetwarzają tę scenę.
Nadal sprawdza tożsamość modeli, zawartość ich plików, vocabulary, ukończenie
wyników i pokrycie progów przez zachowane dowody. Zgodna zmiana batch size,
workerów lub progów publikacji nie wymaga ponownej inferencji.

Po regeneracji VTT/sprite'ów pod tymi samymi URL wymuś weryfikację:

```powershell
python .\stash_ai.py --force-fetch-vtt
```

Switch pobiera i weryfikuje VTT oraz używane sprite'y również dla gotowych scen.
Nie wymusza inferencji: niezmienione, zgodne wyniki zostają użyte ponownie.
Przy zmianie wejść, modelu lub zbyt małym pokryciu cache przeliczane są tylko
wymagające tego wyniki. Opcja dotyczy zwykłej inferencji/wznowienia, nie raportów,
`--merge-only` ani osobnych poleceń rethreshold.

Nowe/niekompletne wyniki i checkpointy nadal wymagają pobrania wejść w przejściu
danego modelu. Na przykład po przerwaniu ukończonego JoyTaga i częściowego Oppai
JoyTag pomija pobieranie, a Oppai pobiera wejścia i wznawia swój checkpoint.
Jeśli ten późniejszy model wykryje zmianę VTT/sprite'ów, wcześniejsze wyniki na
starych wejściach zostają od razu zablokowane i odświeżone przed wspólnym merge.
Blokada pozostaje w bazie po przerwaniu lub błędzie. Wyniki już zapisane na
różnych wejściach, oczekująca weryfikacja, zmiana śledzonych ścieżek lub limitu
próbkowania również wymagają sprawdzenia.

Nie jest potrzebna migracja bazy ani zmiana formatów inferencji/checkpointów.
Istniejące wpisy tożsamości v27 działają od razu przy domyślnym pełnym próbkowaniu.
Starsze wpisy nie śledzą ścieżek; po ich zmianie użyj `--force-fetch-vtt`.
Brak wpisu tożsamości lub starszy wpis przy ograniczonym próbkowaniu wymaga
weryfikacji wejść, po której zgodne wyniki można zachować. Nowe/zweryfikowane wpisy
zapisują mały checksum i kontekst wejść, bez przepisywania surowych wyników.
Historia uruchomień opisuje pominięcie pobierania przez `input_fetch_skipped`;
nie udaje ponownej weryfikacji i nie stanowi podstawy reuse.

### Offline schema updates and repairs

`--db-update` checks schema shape and completed data revisions first. An already current,
complete database returns without decoding inference BLOBs or running SQLite quick_check.
The small metadata/key scan can grow with the number of results. Required historical
conversions (minimum schema v18) run automatically before their completion is recorded.
A previously unrecorded data revision may need a one-time full pass. A DDL-only upgrade
with completed data revisions does not require a global repair.

Use `--db-repair` for an explicit full scan on the current schema: it repairs derived
metadata and identical scene aggregate duplicates, preserving archived/publication proofs.
Conflicting duplicates stop and roll back repair. Semantic corruption should be diagnosed
with `--db-check`; a fast preflight cannot prove arbitrary BLOB contents. Both update and
repair are offline and support `--dry-run`; repair preview uses a temporary SQLite copy.
Neither command runs models, contacts Stash, or globally requests reinference.

Repair spools canonical BLOB rewrites and signature replacements into a disk-backed
temporary SQLite database with a bounded page cache. It scans each original archive
record once and applies evidence, metadata and publication-reference changes in one
transaction; interruption before commit rolls back the entire repair. Memory still needs to fit one
decoded evidence/publication record. Temporary disk space must fit the rewritten BLOBs.

CLI maintenance shows the active phase and scanned/written row counts. A current-schema
update that restores missing metadata or completes a required data revision reports
maintenance performed, rather than claiming that nothing needed doing. A complete
no-op does not create a progress bar or start a full scan.
