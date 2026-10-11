# Facial recognition

The recognition package is independent of the tag model registry and configuration.
The internal layers provide configuration, verified pinned HF assets,
CPU YuNet/SFace adapters, bounded VTT extraction and checkpointed face SQLite.
The public `recognition run` command is introduced by
the subsequent orchestration change; no public scan/cluster/review/apply commands
are provided.

Install the optional standalone dependencies on Python 3.14.7:

```powershell
uv pip install -r requirements-recognition.txt -c constraints/recognition-py314.txt
```

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
