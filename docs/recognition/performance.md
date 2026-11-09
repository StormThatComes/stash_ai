# Recognition execution and recovery

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

The shipped `facial_recognition.yml` is a starting profile for 16 available CPU
slots, 64 GiB RAM allocated to this project and an SSD, with CUDA inference.
It uses 16 rebuild workers, one native thread per worker, a 40 GiB working
maximum and an 8 GiB evidence cache (including up to 4 GiB of matrices).
The 4 GiB sprite cache and untracked process allocations need additional RAM.
Rebuild profiling is initially enabled; disable it after measuring your workload.
The smaller built-in defaults described below still apply to omitted settings.
For four simultaneous solver trials, use the separate
[solver resource profile](solver-parallel.md#resource-profiles).

The run profile uses eight scan CPU workers and a recognizer microbatch of eight.
Static-one ONNX graphs still execute one embedding at a time. Dynamic batching
can slightly change floating-point scores, so check threshold-boundary behavior
with the actual recognizer/provider. The GPU policy is `phase`; model-specific
person/pose/body batches remain unchanged. Neither the YAML nor a 300 GB SSD
allocation guarantees capacity for all retained evidence from 10,000 scenes.

- [Progress and stage counters](#progress-and-stage-counters)
- [Recognition GPU memory](#recognition-gpu-memory)
- [Rebuild performance diagnostics](#rebuild-performance-diagnostics)
- [CPU workers during identity rebuild](#cpu-workers-during-identity-rebuild)
- [Replay memory and checkpoints](#replay-memory-and-checkpoints)
- [Indexed dependencies and certified warm rebuilds](#indexed-dependencies-and-certified-warm-rebuilds)
- [Run-wide immutable evidence cache](#run-wide-immutable-evidence-cache)
- [Independent scene cohorts and late ranking batches](#independent-scene-cohorts-and-late-ranking-batches)
- [VTT CPU pipeline and shared sprites](#vtt-cpu-pipeline-and-shared-sprites)

## Progress and stage counters

Recognition uses the same console renderer as tagging. Terminals show a scene
bar and a second bar for VTT frames in the active scene, with elapsed time,
throughput and estimated remaining time. Publication has its own scene bar;
names and portraits have a performer bar. ETAs describe the current stage,
not the whole invocation, and remain unavailable until work has completed.
The scene bar counts fresh, resumed and reused scans separately. It omits a blended ETA for different kinds of work. The frame bar starts at the saved checkpoint cursor; cached scenes show all
frames reused without treating them as newly inferred work. Frame progress
updates independently of the configured SQLite checkpoint interval.

Identity rebuild has separate progress for publication reconciliation, copying
selected evidence, matching scenes and their face observations, binding historical
identities, saving memberships/candidates, and collecting results. A large scene
therefore advances its observation counter before the scene finishes. Cached
rebuilds show validation work instead of claiming that matching ran again.
The apply bar counts one transaction and completes only after its commit; inner
record counters describe preparation inside that transaction.

Scene selection, model asset verification, name preflight/starter reads and
portrait candidate checks also expose work counters. VTT portraits count ranked
observations; video portraits count candidate frame checks. Known totals show
stage throughput and ETA after completed work is observed. Sources without a
known total, such as paginated all-scene discovery, remain indeterminate and show
completed units and elapsed time. Nested detail bars are terminal-only; redirected
output keeps bounded status heartbeats and major-stage milestones.

In auto and compact mode redirected output keeps phase milestones and one status heartbeat per minute,
including scene/frame counts and stage ETAs, without per-scene frame logs.
`--progress-style verbose` adds per-scene quality/reference details and the
performer names YAML starter. Compact progress skips optional SQL counts and uses known totals for a rolling stage ETA. The last summary includes total command time and the three longest phases. The starter remains recorded in face SQLite
history in every style. `--progress-style off` keeps errors and the final
summary, including aggregate rejection reasons and reference usage. Repeated
scan/publication errors are grouped; failed scenes still count as inspected
work and are reported separately from successful scans and reused evidence.

Compact progress shows process CPU in cores, current RSS and disk figures. The run displays the last
per-volume budget snapshot (charged files, outstanding reservations and free space). When no budget
snapshot exists, the disk line names the source SQLite files and their volume. Refreshing the display
does not enumerate the cache. A current wait label identifies VTT prefetch, SQLite checkpoint writes,
ordered CPU work or HTTP backoff.

## Recognition GPU memory

```yaml
version: 2
execution:
  gpu:
    model_policy: phase
    log_vram: true
```

`phase` is the default for existing YAML files too. Models remain shared during
the VTT phase, then unload before rebuild. `sequential` additionally separates
face detection/embedding from person evidence collection **within each scene**:
faces finish and unload before person models load; person models close before
the next scene. It trades model reloads and a second read of verified VTT crops
for lower concurrent model residency. Completed face scans and person evidence
remain reusable after interruptions; a failed person pass does not publish that
scene. Switching policies does not invalidate evidence or change thresholds.
Without SapiensID, `sequential` behaves like `phase`.

`sequential` does not promise one model at a time: the person pass still holds
the selected person detector, pose estimator, optional SAM and SapiensID encoder
together. Batch sizes still control temporary inference allocations. `retain`
keeps face/visibility models through rebuild for the legacy scan/portrait reuse
behavior; person models still close after VTT. All modes reclaim unused cache at
run exit, including failures, after workers relinquish their resources.

`log_vram` shows the latest phase measurement and sampled maximum on one terminal row. A missing measurement is reported once in compact output. Verbose output includes boundary samples. The option writes phase snapshots and bounded `gpu_memory` run history. PyTorch
`allocated` measures live tensors; `reserved` includes its reusable cache. If
`nvidia-smi` is available, `process` sums this PID's compute-process usage across
NVIDIA GPUs, including ONNX allocations. Missing/unsupported values (including
WDDM `N/A`) are reported as unavailable, not zero. PyTorch queries cover its
current CUDA device and never initialize CUDA or import PyTorch for monitoring.
Sampled peaks are boundary observations, not the true intra-inference peak.
Scene-level release lines are verbose-only. At most one new scene is sampled
every ten seconds (keeping its before/after pair), and history retains the latest
scene per boundary rather than an unbounded event list. Set `log_vram: false` to disable
queries and logs without disabling cleanup. VRAM is separate from
`execution.rebuild.memory_bytes`, which budgets CPU working allocations.

The VTT scan shares its pixel worker pool between face and person preparation.
When CPU workers and face microbatches are enabled, preparation from one
detector batch can run during the next detection. File-backed scans can keep
one complete checkpoint transaction in flight. Recognition reports a durable
cursor after its face rows, pose evidence and shot state commit together.

To compare SapiensID batches on the same labelled scenes and configured pose
backend, run the following from the repository root.

```shell
python -m benchmarks.benchmark_crowded_vtt --config facial_recognition.yml --labels labels.json --body-batches 1 2 4 --cuda-profile
```

Compare actual forward batch sizes, bodies per second, peak CUDA allocation
and assignment quality before choosing the encoder batch size.

`execution.scan.prefetch_frames` defaults to 4 ready thumbnail crops. A CPU producer
decodes/crops images ahead of inference, preserves cue order and owns independent
copies until consumed. The queue retains at most that many ready crops, plus the
producer and consumer's in-flight frames; `0` uses synchronous frame preparation.
`execution.scan.prefetch_scenes` defaults to `true` and verifies/downloads upcoming
selected scenes while the current scene scans. `execution.scan.prefetch_scene_depth`
controls how many scenes are prepared ahead (default 3, range 1–16), with at most
four input workers. Each input owns isolated read-only HTTP sessions. Set
`prefetch_scenes` to `false` to disable lookahead independently of frame preparation.

Both options leave model/scan identities unchanged, so existing observations are
reused. Producers never run models, write SQLite or publish Stash mutations.
Queued frames are speculative until inference completes: failures/interruption
save only completed cues, cancel media reads and close queued images/readers.

Recognition has no byte ceiling on VTT/sprite inputs, no fixed VTT cue-count
or sprite-sheet count ceiling, and no pixel ceiling on decoded images or detector
input. Optional `execution.scan.max_cues` still selects an evenly distributed sample; `0` processes
all cues in the selected scene. The retired `max_vtt_bytes`, `max_media_bytes`,
`max_image_pixels` and `max_sheets` execution keys have been removed from the
configuration and bundled template. Remove them from existing local YAML before running;
unknown keys are rejected rather than ignored.

Tagging and recognition share a Pillow reader that disables decompression-bomb
pixel thresholds for Stash images, including subsequent crop checks. Large-image
warnings no longer clutter output; other warnings and image/rectangle validation
remain active. Video portraits use native frame dimensions without pixel-driven
downscaling.

Scene performer associations use Stash's `BulkUpdateIds.ids` with modes `ADD`/`REMOVE`,
preserving unrelated performers. On the next explicit run, a legacy unconfirmed
link with the old malformed input and its exact recorded HTTP 400 can recover
after fresh ownership and scene reads; new intents record their input contract.
Timeouts, connection failures, other HTTP errors and unrelated uncertain writes
remain protected from blind retries. HTTP 400 GraphQL validation messages are
reported with bounded credential-redacted detail instead of only the endpoint.

For schema upgrades, backup ownership and interrupted histories, see
[recognition state](storage.md#schema-and-backups).

Face completeness/pose currently use measurable geometry and five-landmark
ordering/symmetry estimates. They do not prove that every feature is unoccluded.
Sharpness is measured on a normalized grayscale crop; actual face dimensions,
exposure and clipping are retained separately.

## Rebuild performance diagnostics

Set `execution.rebuild.profile: true` to save bounded `rebuild_profile` aggregates
in recognition run history, including failed/interrupted attempts. The default
is false. Measurements contain no SQL text, scene/person IDs, images or vectors.
They cover copy, reference loading, scene matching, contextual checks, merge
support/compatibility, rechecks, binding and final apply. Work counters include
validated vectors and group/sample comparisons. SQL buckets count statement
execution and transaction exits; statement seconds exclude later lazy cursor
consumption. Stage seconds are inclusive and nested, so do not add them together.
Record runtime, observation/person counts and warm versus cold state when
comparing runs. Profiling has overhead and does not alter matching policy or
require fresh inference. Ordinary progress remains throttled independently.

Cold replay now owns a RAM-resident canonical reference index for its immutable
selected observation catalogue. Membership inserts, moves and retractions update
that private index; unchanged representatives reuse validated vectors. Known
references remain frozen inside each scene. The exact canonical sample order,
reference fingerprint and logical reference budget are preserved. General
reference loads use bounded person batches, including full membership-scope
validation outside the representative cap.

`execution.rebuild.memory_bytes` is a shared live working budget (8 GiB by default),
covering replay pages, working reservations and evictable caches. It is not an
upfront allocation, an OS RSS cap or the representative-vector limit. It can be
increased on hosts with more RAM, such as a 96 GB system. If the canonical index
cannot fit its conservative per-observation allowance, replay uses the existing
ordered queries without a new disk-spill cache. A failed replay discards its
private index; it is never reused after rollback or by another invocation.

Similarity scoring uses bounded native NumPy blocks for sufficiently large
catalogues. A conservative floating-point error interval may reject only groups
that cannot meet the relevant threshold/cohesion condition. Every remaining
comparison uses the original scalar dot products and reductions, preserving
stored scores, ambiguity, ties and decisions. Small catalogues keep the scalar
path. Native block caches are invocation-owned and byte-bounded, including their
retained input owners; no approximate-nearest-neighbor or truncated rival list is
introduced. Same-scene contextual matching retains its full cohort of rivals.

Cold replay and each standalone scene match own a RAM-only evidence cache. Cue
rows are narrow and vectors/signatures are validated lazily, preserving existing
short-circuit failure behavior. Exact peer scores are reused only for the same
observation and immutable representative version. Contextual merge checks share
verified latest-scan heads and complete multi-face evidence, including withheld
observations and later selected scenes. Newer partial or differently configured
scans still supersede old evidence. The cache uses a bounded share of the working
budget and recomputes evicted data instead of spilling to disk.

## CPU workers during identity rebuild

See [the retained-evidence benchmark and CPU diagnostics guide](benchmarking.md)
for read-only local baseline collection and computed-versus-applied progress.

`execution.rebuild.workers` controls the rebuild compute pool independently of
`portrait_cpu_threads` and the optional one-reader `rebuild_prefetch_scenes`.
The default `1` keeps serial execution. `0` chooses up to sixteen workers,
leaving two available CPUs free where possible. Explicit values from 1 to 256
are capped by available CPUs divided by `rebuild_native_threads` (1..16).
These are logical CPU budgets, not core pinning or guaranteed utilization.

For an initial comparison on a 24-core host, try 1, 8, 16 and 20 workers with
one native BLAS thread per worker. For example:

```yaml
version: 2
execution:
  rebuild:
    workers: 16
    native_threads: 1
    queue_bytes: 67108864
    profile: true
```

This is a tuning starting point, not a measured fastest setting. Worker startup,
Python loops, matrix packing, memory bandwidth and serial decisions can dominate
small workloads. Sufficiently large immutable reference catalogues use bounded
read-only matrix shards and query batches. Small catalogues, mutable inputs and
work that cannot fit admission keep the existing serial score path.

The coordinator still visits scenes and observations in canonical order. It
alone grows new candidate groups, makes merge/support/co-presence decisions,
claims historical person/performer labels and writes SQLite. Workers calculate
frozen-reference scores used by assignment, post-merge rechecks, merge
nominations, final same-scene rankings and historical fallback. Native batch
scores may only prove rejection; finalists retain the scalar decision values.
Demanded component-pair compatibility proofs also use immutable CPU jobs when
multiple constituent pairs are needed. Each proof keeps the directional scalar
dot/mean/min order; the coordinator validates reference ownership and imports
both positive and negative proofs into the existing bounded cache. A small
batch is retired before an incompatible endpoint stops later work. No global
all-pairs identity graph is scheduled.

For longer demanded support/context streams, bounded CPU blocks compute exact
constituent predicates. The coordinator still selects the authoritative latest
scans, reduces votes in scan/cue/observation order, counts each scene/time once
and applies distinct-signature co-occurrence rules. Mutable direct membership
is checked at decision time before contextual work. A complete small batch is
retired before exposing its ordered results, allowing early exit without
abandoning an active CPU stream. Invalid vectors keep their original point of
failure, and small/oversized streams use the scalar path. These ephemeral masks
are not persisted or reused across component changes; shared versioned mask
caching remains separate work. Cue-peer/rival checks remain on the coordinator.

Rebuild uses retained embeddings and never loads or invokes GPU models. CPU
workers receive neither models nor SQLite connections. Scan and portrait inference remains separate from this CPU pool;
model release follows the GPU policy described above.

`rebuild_queue_bytes` bounds queued CPU jobs including estimated scratch/results;
its effective value is capped at the admitted `rebuild_memory_bytes`. Packed matrix
caches, the bounded current query batch and sparse-result accumulator retain
their separate bounded ownership. This field is not a total-process RSS cap or
unified accounting of every rebuild cache. No disk-backed working cache is
introduced. Native BLAS limits are restored only after their workers stop;
incomplete shutdown retains borrowed owners under the existing quarantine rule.

With profiling enabled, `rebuild_profile.cpu` records requested/effective workers,
native limits, queue capacity/peak, submitted/applied blocks and ordered wait.
Successful worker wall seconds are summed parallel task durations, not elapsed
run time. Worker-thread CPU seconds exclude native helper-thread CPU time; the
benchmark additionally records process CPU time. Worker metrics do not retain
scene IDs, embeddings or SQL. Progress remains based on coordinator-applied
scenes, so prefetched computations cannot advance the bar early.

Execution budgets do not change matching policy or semantic checkpoint identity.
A restored checkpoint resumes only the applied strict prefix and can use a
different worker count. Changing workers on an already valid completed rebuild
can reuse the result, so it does not measure cold-rebuild scaling.

Run the synthetic cold/cached comparison independently of production state:

```powershell
python -m benchmarks.benchmark_recognition_parallel --scenes 480 --people 96 --dimensions 512 --workers 1 8 16 20 --repeats 3
```

The fixture contains repeated orthogonal identities and omits model inference,
media, API latency and realistic face/merge distributions. Its timings are not a
forecast for a 9,000-scene library. The JSON includes semantic hashes, effective
budgets, stage profiles, process CPU and RSS after each trial. Use representative
retained evidence and inspect the limiting stage before promoting a default.

## Replay memory and checkpoints

Replay proof caches reuse exact scores only while immutable representative
versions remain live. Directional constituent compatibility (including rejected
pairs) and earliest distinct support-cue prefixes share the byte-bounded RAM
owner. Every changed representative set, threshold or support minimum causes
revalidation; evicted entries are recomputed. Co-presence still includes all
eligible selected scenes, and membership checks remain current. Weak input
owners prevent caches from keeping obsolete reference sets across reloads.

Cold replay batches historical ownership reads and prepares a single ordered
label-vote stream with frozen alias/performer metadata. A private, transactionally
maintained candidate-address relation avoids repeated payload decoding in hot
joins; durable payloads and their integrity checks keep the existing format.
Physical sample keys use a bounded RAM cache keyed by every canonical dependency,
so changed geometry, source or scan input cannot reuse an old key. Final candidate
payloads receive their reference context during insertion rather than a second
JSON update pass.

The disposable replay database now resides in RAM, including SQLite temporary
work. Its live main-database pages share `execution.rebuild.memory_bytes` with
working reservations and the evictable evidence cache. There is no fixed quarter
for Python work or half for SQLite: each can borrow unused capacity. Page ceilings
shrink before other owners reserve that capacity. Exceeding the shared budget
aborts the private replay before durable apply, with owner, requested bytes and
reservation totals in the error. Model memory, transient SQLite sorts and other
untracked overhead are additional allocations; this is not a process RSS cap.
Durable FaceDatabase transactions retain their existing guarantees.

`execution.rebuild.prefetch_scenes` defaults to false. This experimental option
uses one isolated read-only reader to validate at most one upcoming scene.
There are no fixed 256-observation or 8 MiB scene eligibility limits. Once
`execution.rebuild.prefetch_scenes: true` is selected, prefetch is mandatory for
each selected scene, including the first scene in a cold replay. It prepares
all eligible vectors without a separate per-scene prefetch RAM fraction or
observation limit. Actual allocation failures, SQLite lock/read failures and
worker errors fail the rebuild instead of silently falling back to serial
decoding. Current and pending prefetched vectors reserve space in the same
`execution.rebuild.memory_bytes` budget as replay pages and other working data. Disable the
option explicitly to use serial input handling. The read-only worker traverses indexed observation order
without a large SQLite sort. Only immutable vectors are prepared; assignments
and SQLite writes stay ordered on the coordinator. Invalid vectors are reported
when originally consumed, and cancellation closes the reader/worker.

Run `python -m benchmarks.benchmark_recognition_replay --scenes 960 --repeats 2`
for an offline comparison with the former temporary-file path. One Linux run
(Python 3.14.7, NumPy 2.5.3, SQLite 3.53.1; 2,880 observations, six orthogonal
identities, 128 dimensions, four representatives) measured medians 4.55 s file,
4.19 s RAM serial, 4.72 s RAM lookahead, with identical partitions. The replay
image was 5.5 MiB. These synthetic results support keeping serial preparation
default; they do not predict a production rebuild's completion time. The
benchmark reports process RSS after each trial, not isolated peak process RAM.

Matching policy v8 separates representative/decision semantics from the
`max_reference_bytes` and `max_people_per_lookup` admission limits. Existing v7
candidates receive one retained-evidence replay; no new inference is requested.
Completed rebuilds can reuse results after monotonic capacity increases only
when all current input, reference, label-origin, alias and exact candidate
integrity checks pass. Decreases replay conservatively, as does an increase
when historical label matching previously skipped an over-budget reference set.
Standalone scene caches also record and check their original admission limits.

Keep at most eight completed selection descriptors per scope/namespace. A subset
run does not discard a full selection's descriptor, but it can invalidate that
selection's dependencies or assignments. Descriptors contain bounded hashes and
limits, not copied vectors or historical assignments. Current memberships,
accepted/withheld coverage, exact candidate UUID addresses, raw input repairs and
performer uniqueness are revalidated. Broken derived descriptors/candidates fall
back to retained-evidence replay. Warm checks read 256-scan chunks and validate
homogeneous vector blocks of at most 128 rows/1 MiB; norm-boundary and malformed
records use the original scalar validator for exact error behavior.

Interrupted cold rebuilds can optionally keep durable recovery generations with
`execution.rebuild.checkpoint_scenes` (default **0**, disabled; for example **256**).
Active replay, temporary work and caches remain in RAM. The sidecar directory
`<recognition-database>.rebuild` stores only derived strict grouping/candidate,
alias and merge-support rows, without observations, vectors, JPEGs or old labels.
`execution.rebuild.checkpoint_bytes` caps each generation (default 512 MiB,
maximum 8 GiB); publication briefly needs both old and new generations. Raw
selected evidence and current historical labels are copied again on resume.

Resume validates directory/database ownership, engine/schema versions, exact
canonical selection, the **entire** previous selected input catalogue (including
unprocessed future scenes), matching policy, admission limits, checksum, cursor,
exact next identity clock and accepted/withheld/support/address invariants. It
then skips only a fully committed strict prefix. Same-scene context runs once
after all strict scenes; label binding and durable assignments still apply in
one final transaction. Progress starts at the restored scene cursor, and ETA
uses only new work. No inference or remote publication is requested to restore.

Immutable generation files are fsynced before an atomic manifest publication;
failed/cancelled writes preserve the previous generation. Owned stale generations
are pruned, foreign files/symlinks are refused, and invalid derived state falls
back to full retained-evidence replay. Optional checkpoint I/O/capacity failure
is reported once and disables saving for that run without failing recognition.
A successful strict generation remains available after final apply.

Append-prefix reuse is deliberately narrow: the old full canonical selection
must be an unchanged prefix, policy/engine and old raw inputs must match, admission
limits cannot decrease, and every added cue must contain at most one distinct
physical face signature. Exact duplicate detections are allowed. This proves
added contextual evidence cannot veto an earlier merge; any distinct co-present
faces, inserted/replaced scenes or uncertain dependency cause full replay. All
future selected contextual evidence remains visible to the original matcher.
Checkpointing prevents repeated work after interruption; it does not promise a
faster uninterrupted first run. Run
`python -m benchmarks.benchmark_replay_checkpoints --scenes 960 --repeats 2`
to measure full, checkpointed, resumed and safely appended synthetic selections.

An offline Linux probe (Python 3.14.7, NumPy 2.5.3, SQLite 3.53.1;
960 scenes / 2,880 observations / six orthogonal identities / 128 dimensions /
four representatives; two trials, interval 240) measured medians: full 3.83 s,
checkpointed 4.05 s, resume after 720 scenes 0.84 s, safe append from 720 scenes
0.92 s. Partitions and exact candidate scores matched across all modes. The
retained generation was 2.01 MiB; the new benchmark samples process RSS every
10 ms. These are complete synthetic replay timings without models/network and
include input copying, dependency verification, binding and final apply. They
are not estimates for the operator's 9,000-scene workload.

## Indexed dependencies and certified warm rebuilds

Schema version 5 adds durable per-scan/global evidence versions and bounded
fingerprint records. Migration from versions 1–4 preserves observations,
identities, installation IDs and publication history. Candidate observation
addresses use a TEXT expression index: the cast matches the affinity of
`observations.id`, allowing SQLite to seek the latest decision instead of
scanning all candidate JSON for each observation. The actual observation address
wins over a candidate's claimed scan or policy.

Fixed application-owned triggers update dependency versions in the same
transaction as evidence changes. Rollback restores both evidence and versions.
Separate BEFORE INSERT guards handle SQLite REPLACE with recursive triggers off,
invalidating both old and new observation addresses. Unknown, modified or missing
required triggers fail schema validation; database-provided SQL is never executed
to repair the schema. JPEG-only changes do not invalidate identity evidence.

Selected-scene input digests are cached independently. Global history and label
binding digests retain conservative dependency stamps. Completed rebuilds receive
a certificate covering the full descriptor and current evidence version. An
unchanged, admissible descriptor can reuse decisions without rereading all
vectors; missing/altered certification takes the original bounded integrity path.
Candidate, membership, identity, source geometry, quality, scan and model changes
invalidate reuse. Arbitrary edits to a completed metadata descriptor cannot
reuse its previous certificate.

Caches preserve caller transactions and readonly evaluation remains readonly.
Each scene/policy cache retains at most eight variants; historical selection
contexts retain at most 128 records. Profiling exposes `dependency_digest_hits`,
`dependency_digest_misses` and `certified_warm_rebuilds`. Cold validation still
streams evidence and reports observation progress; warm validation reports the
selected-scene work without pretending to reread embeddings.


## Run-wide immutable evidence cache

`execution.rebuild.cache_bytes` (default 1 GiB, capped by currently unused
`rebuild_memory_bytes`) shares one LRU budget across validated vectors, immutable
JSON metadata, sealed replay cue evidence, packed matrices, numerical pair evidence
and lazy co-presence lookups. `rebuild_matrix_cache_bytes` (default 512 MiB) caps
all retained matrices within that same budget. Zero disables either cache. These
execution settings do not change the matching policy or retained decisions.

Vector/matrix/pair keys contain exact immutable float32 bytes. Reconstructed banks
can reuse packed matrices; mutable or sliced arrays bypass this path. Pair cache
entries contain only original scalar numerical reductions: current support,
authority, thresholds, rivals and ownership still decide whether a merge is allowed.
Co-presence entries include the live connection, dependency generation, local write
counter and external data version; a membership edit or rollback cannot reuse a
stale result. Unsupported legacy replay schemas use the original SQL path.

The cache covers all rebuild phases and is discarded at run exit, including errors.
Eviction/oversize entries use the exact uncached computation. Keys, source byte
owners and decoded metadata are included in conservative reservations. Profiling
reports `ram_<category>_hits`, `_misses`, `_oversize`, `ram_evictions` and
`ram_cache_peak_bytes`/`ram_cache_limit_bytes`; this is a cache-retention budget, not
a process RSS ceiling (SQLite, active reference banks and worker scratch are separate).


## Independent scene cohorts and late ranking batches

With `execution.rebuild.workers > 1`, independent scene-local appearance priorities
are calculated concurrently. Each job receives only plain selected observation data
and matching configuration, uses a private in-memory staging database and returns
observation priorities. The coordinator applies those priorities in selected scene
order. No worker assigns identities, writes the authoritative database or owns a GPU
model. Cue times, independent support and frozen reassessment remain unchanged.

`execution.rebuild.cohort_backend: threads` is the default; `processes` selects
spawned processes for Python-heavy cohort loops and keeps native BLAS thread limits.
This option affects cohort jobs only; existing numerical score jobs retain threads.
Scene job reservations include serialization, staging SQLite, witness arrays and
results under the existing CPU queue budget. A scene that cannot fit is streamed
on the coordinator instead of being truncated. Cancellation propagates to workers
and all workers are stopped before releasing their owners.

Late strict, historical, appearance and uncertain-pose queries reuse one prepared
reference catalogue per bounded batch. CPU jobs return original scalar scores for
all rivals; the existing coordinator ranking/screening supplies the same proofs,
including weak rival scores and ties. Fallback appearance results still cannot
train another match. Profiling adds `local_cohort_scenes` and `process_cohort_blocks`.


## VTT CPU pipeline and shared sprites

`execution.scan.cpu_workers` (default 4) prepares face quality, supported ONNX
alignment tensors and JPEG crops on CPU workers. Detector, visibility and embedding
inference remain on the calling model owner. Preparation and encoding use bounded
queues (`scan_queue_bytes`, default 256 MiB); workers own their image copies and
oversize jobs run synchronously. The queue allowance reserves estimated image
scratch and results; it does not limit input image dimensions or source downloads.
A cue advances its checkpoint only after every face finishes. Any error discards
that cue's tentative observations and retains earlier complete cues.

`recognizer_batch_size` defaults to 1. Values up to 32 enable AuraFace/LVFace/AdaFace
microbatches only if both ONNX input and output batch dimensions are dynamic.
Static-one graphs use exact single-image calls; unsupported backends keep `embed`.
GPU batch sizes can change floating-point results slightly: benchmark/calibrate
an opt-in batch on the actual model/provider before relying on boundary scores.

`prefetch_scene_depth` defaults to 3 (1–16), with at most four separate input
readers. Results are consumed in requested scene order, errors stay associated
with their scene, and all speculative inputs are closed on cancellation.
`prefetch_scenes: false` retains synchronous input loading.

`execution.scan.sprite_cache_bytes` defaults to 1 GiB. One thread-safe, content-addressed
LRU shares compressed sprite bytes and decoded RGB sheets across scan and portrait
input owners. Source manifests still verify their own sprite hashes. Cropping
holds the cache lock so eviction cannot close borrowed pixels; only owned crops
leave it. Zero disables retention and oversized sheets use crop-before-conversion.
The cache ends with the invocation, including failures, and does not cache credentials
or identity decisions. Active decoded crops and worker scratch are separate from
this retained-sprite allowance.

Portrait preparation now uses multiple CPU jobs for supported FaceOcc input
preparation and frozen identity ranking. Candidate acceptance, SQLite updates and
final selection remain on the coordinator; one model executor owns all portrait
inference. `portrait_cpu_threads` bounds the CPU preparation workers (automatic
when zero); the existing JPEG owner can overlap preparation for the next target.
All original observations remain eligible for ranking, including after a crop fails.

For the stacked recovery and throughput changes, see [recognition-sixth-series.md](../archive/recognition-sixth-series.md).


Publication validates a frozen scene certificate before each guarded action.
Repeated checks reuse its vector digest only while audited scene generations,
current scan, identity ownership/tokens and body addresses still match. Profile
binding and publication journal writes do not rescan every embedding. Evidence
changes, writes through another database connection or executable schema changes
require fresh validation; a certificate is never advanced after a changed or
rolled-back generation.

The shipped profiles use `execution.rebuild.checkpoint_mode: incremental` for
strict local replay. The first checkpoint stores a base; later checkpoints store
only transactionally changed rows and deletions. Immutable segments are published
by an atomic manifest switch and verified before import. Every 16 generations a
new base compacts the chain. A failed write retains the preceding committed
prefix. Late/ready phase checkpoints are still independent full images.

`checkpoint_bytes` bounds the current strict chain (the shipped profiles allow
8 GiB); old/new generations can coexist during publication, and the shared
per-volume disk budget includes that temporary overlap. `snapshot` retains the
previous full-prefix format. Large benchmarks remain manual.

The raw replay input cache uses selected raw-observation and pixel dependency
stamps, rather than the global decision clock. It contains no frozen ownership:
performer labels and historical origins are refreshed from authoritative SQLite
before matching, including on a cache hit. Its byte limit is
`replay_snapshot_bytes`, independent of the derived checkpoint size.

Independent strict face scenes can run as complete CPU jobs with
`execution.rebuild.parallel_scenes` (default true). It requires anchor/view
matching and more than one rebuild worker. Each worker owns a private database;
only the coordinator publishes decisions in canonical scene order. Queue admission
includes a conservative scene working allowance; oversized scenes use the serial
path. Threads and spawned processes share this contract. Body crop quality also
uses the bounded scan CPU stage, without moving model or SQLite ownership.

The global reference index is allocated only on first use after independent local
matching. Canonical observation UUIDs, membership ranks, flags and interned owners
use packed storage; insufficient working memory retains the SQL fallback.

Body appearance support uses an exact Hamming-radius tree. It keeps the first
inserted representative, including strict distance boundaries, so repeated pixels
and cue votes retain their previous meaning. Co-presence rival checks use bounded
score screening and stop only after an unchanged scalar score proves separation.
The optional hybrid cue index spills to budgeted scratch SQLite when its RAM
allowance is exhausted; exhausted scratch falls back to the original exact SQL.

`execution.scan.inference_cache_bytes` enables a disposable SQLite cache next to
the recognition database (shipped: 8 GiB; legacy/default configuration: disabled).
Detection, top-down pose, prepared face/body embeddings, and measured quality
have separate versioned keys. Quality/identity thresholds are reapplied to cached
measurements. A key includes exact pixels or prepared tensors, geometry, model
checksums, active precision/device and runtime versions. Changing a body encoder
therefore does not rerun pose inference for the same image and boxes.

Identical inputs share inference, including within a microbatch. Every cue still
produces its own observation, timestamp, geometry contract and independent-evidence
vote; cache entries never store assignments or support counts. Values are copied
on reuse. Checksums reject corrupt entries, payload/page limits bound storage,
and cold inference remains available after eviction. Very small budgets below
64 KiB disable the optional cache. The file contains arrays/JSON, never pickle.

RF-DETR, ViTPose and prepared SapiensID inference retry CUDA OOM by halving only
the failed batch, preserving order and remembering the smaller size for the model
lifetime. A single-item OOM fails explicitly; CPU errors and other kernel errors
are never retried or converted to CPU inference. Retry counts appear in pipeline
profiling and each capacity reduction is logged.

CUDA inputs use pinned host tensors with nonblocking transfers. ViTPose heatmaps
use a dedicated copy stream and an event before CPU decoding; one previous
readback can overlap the current inference. Pending readbacks are drained on
success, error and close. CPU/MPS retain their ordinary transfer path. The small
CI fixtures simulate OOM and transfer events without a GPU. Measure throughput,
peak VRAM and output tolerance on a real CUDA machine with the manual benchmark;
no full 10,000-scene inference is part of CI.

During recognition, the SSD line names the volume with the least remaining capacity under either the data cap or the free-space reserve, including the age of that measurement. VTT prefetch reports ready inputs over all pickups and time actually spent waiting. CUDA OOM recovery reports the failed batch size, the reduced size and retry count even with VRAM logging disabled. Process VRAM comes from a rate-limited background query and can be unavailable on some drivers. Rejection counts for an unchanged scan reuse its retained evidence revisions and active face/body contracts.
