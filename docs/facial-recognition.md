# Facial recognition

The recognition package is independent of the tag model registry and configuration.
The internal layers provide configuration, verified pinned HF assets,
CPU YuNet/SFace adapters, bounded VTT extraction and checkpointed face SQLite.
The only public operation is the explicit, finite `recognition run` command.
No public scan/cluster/review/apply commands are provided.

Install the optional standalone dependencies on Python 3.14.7:

```powershell
uv pip install -r requirements-recognition.txt -c constraints/recognition-py314.txt
```

With `STASH_URL` and optional `STASH_API_KEY` set for Stash 0.31.1 or later:

```powershell
python stash_ai.py recognition run --scene-id 6306
python stash_ai.py recognition run --scene-id 6306 --scene-id 6307 --face-config facial_recognition.yml --faces-db faces_state.sqlite3
python stash_ai.py recognition run --all-scenes
```

The last command explicitly snapshots all scene IDs for this one invocation.
There is no watcher, tagging hook or background recognition process. Ordinary
tagging, merge, cleanup and `--models all` neither start recognition nor read its
YAML. Missing pinned weights are installed automatically on recognition runs.
Eligible identities become anonymous performers and receive additive scene
links. Weak identities remain recorded with reasons and do not block clear ones.

Ownership uses an exact installation/person token in performer details, never a
name-only match. Each mutation intent is committed to the face SQLite before
dispatch. Lost creation/link responses are reconciled by fresh Stash reads;
unresolved outcomes are withheld without blindly retrying. Existing manual links
are preserved with Stash's `ADD` mutation. Selected-scene runs do not resume links
for other scenes. Exit 0 means the finite selection completed; exit 1 records
partial/failed work, and interruption returns 130 from the root launcher.

`facial_recognition.yml` is a versioned face-only configuration. Numeric matching
thresholds are cosine similarities, not calibrated probabilities. The initial
0.65 match/0.60 cohesion defaults prioritize withholding uncertainty; they do not
claim measured accuracy on a user's VTT library. OpenCV's example LFW threshold
is not used as a Stash-thumbnail accuracy guarantee.

All relative model, database and names paths resolve against the selected YAML
directory. Missing default YAML uses bundled defaults; an explicitly selected
missing file fails. Values are immutable during an invocation. The default
`faces_state.sqlite3` is reserved for face information and is independent of
tagger state/history. A names/portrait/threshold setting is not a raw model identity.

Official OpenCV Python wheels run these adapters on CPU. `device: auto` reports
CPU, and an explicit unsupported `cuda` request fails instead of claiming GPU
execution. Weights are checked against immutable revision-bound SHA256 values;
existing custom/corrupt files are never silently overwritten.

Normal core tests do not download models. The separate recognition CPU CI job
loads the pinned detector/recognizer on Linux and Windows. This is a runtime,
alignment and embedding-shape smoke test, not an accuracy benchmark.

## Internal media and state

Each selected scene verifies the VTT/sprite content before reusing observations.
Sprite and decoded-pixel limits apply before conversion, with at most two active
sheets. Completed cue checkpoints survive an interrupted inference; incomplete
scans cannot publish identities. Partial sampling and rejected/no-face outcomes
remain explicit coverage information, not proof that a person is absent.

One face SQLite stores model/scan provenance, detections, normalized embeddings,
bounded crop BLOBs, accepted identities, Stash ownership/bindings, candidate
portraits, mutation records, progress and run history. Its independent schema
refuses tag/history/foreign databases and hard-linked aliases. Internal backup
uses SQLite's consistent backup API; no result/history/image sidecars are needed.

Face completeness/pose currently use measurable geometry and five-landmark
ordering/symmetry estimates. They do not prove that every feature is unoccluded.
Sharpness is measured on a normalized grayscale crop; actual face dimensions,
exposure and clipping are retained separately.

## Automatic identity policy

Matching is incremental: selected-scene vectors can reuse compatible retained
people from other scenes without fetching their media or rewriting their scene
associations. New identities need the configured distinct cue-time support.
Insufficient, near-threshold or ambiguous matches are withheld automatically.

The best and next coherent identity compete even when the runner-up falls just
below the match threshold. Cohesion is checked against bounded representative
samples, avoiding transitive single-link unions. This sampled check is not a
proof that every historical pair matches. Reference bytes, samples and identity
counts have explicit limits; no N-by-N distance matrix is allocated.

Developer-only `evaluate_pairs` accepts explicit local same/different-person
ground truth and reports false matches/misses for that supplied dataset. Synthetic
vector fixtures prove grouping rules, not real-world face-recognition accuracy.
The default policy is labeled uncalibrated for a user's domain in stored provenance.

## Automatic portraits and operator names

An eligible person without an image receives the best evaluated identity-verified
video-frame crop. Install system FFmpeg (or configure `portraits.ffmpeg`); absence
or unsupported/budget-limited video is recorded as an image skip while links
remain eligible. Existing Stash images are preserved. The baseline missing-image
contract is its own `/performer/ID/image?...&default=true` URL, not a guessed
placeholder filename. A fresh check immediately before the mutation also protects
images supplied during extraction. Stash has no atomic image/name compare-and-set
mutation: a server-side edit between this check and dispatch remains a race.

The shortlist alternates selected scenes and spreads cue times. At most
`max_candidates` observations times `time_offsets` are evaluated; each selected
scene decoder shares one byte/time budget across its seeks. Only selected-scene
video is fetched, even for a historically known person. FFmpeg receives an owned
loopback range URL, not Stash credentials or signed media URLs. Authenticated
same-origin requests, byte/deadline limits, process/log/output limits and downscale-only
pixel limits apply. Actual presentation times, identity comparisons, quality,
exclusions, crop/margin, policy and final bounded JPEG BLOBs stay in face SQLite.
The best evaluated crop under these budgets is not a global-best-frame claim.
Visibility remains the documented five-landmark estimate, not semantic occlusion
detection. Failed decoding may be retried on a later explicit run.

Configure `names.file: performer_names.yml` to opt into this separate optional file:

```yaml
version: 1
stash_scope: "http://localhost:9999"
performers:
  "123": "Example Performer"
```

Use the exact configured Stash base URL (or the scope digest printed in the bounded
starter). IDs must be quoted canonical positive strings. The complete file is
validated and snapshotted once before Stash/state/model initialization. Within
the run, only mapped accepted/bound selected IDs enter the name plan; all selected
targets are resolved before any name updates. An unknown selected target prevents
the whole name batch; unrelated mapping entries are not resolved or changed.
Updates change only the name, with fresh ownership/current-name checks, typed
responses and uncertain-outcome reconciliation. Equal names do not merge IDs.
Both YAML snapshots, resolved names/preconditions and history remain in face SQLite.

Normal output includes at most 50 selected published IDs/current names as a YAML
starter, also stored in SQLite. It never overwrites your mapping file or treats
these generated anonymous placeholders as rename instructions. No additional
command, manual portrait picker or persistent image/operation sidecar is required.
Image acknowledgement recovery compares the submitted JPEG digest to Stash's
served image. A server that transforms the image is treated as an unresolved or
conflicting result and its live image is preserved, rather than silently retried.
