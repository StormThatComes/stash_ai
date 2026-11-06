# Stash AI

Automatic scene tagging, time-based markers and on-demand performer recognition for [Stash](https://github.com/stashapp/stash).

Tagging reads Stash VTT files and sprites, runs JoyTag, WD EVA02-Large v3,
DanbooruTagQuery L/16 and OppaiOracle V1.1 sequentially, and publishes shared
`ai:` tags and markers. Model evidence and publication state stay in a local SQLite database.

Recognition is a separate explicit workflow with its own database and YAML.
It matches face and optional body evidence from VTT cues, creates performers and
scene links, and selects portraits. Video decoding is optional and used only for portraits.

## Requirements

- **Python 3.14.7**, the supported and tested version.
- **Stash 0.31.1 or newer**, with generated VTT files and sprites.
- One application profile: GPU (`requirements.txt` / `requirements-gpu.txt`)
  or CPU (`requirements-cpu.txt`), with matching [constraints](constraints/README.md).
- CUDA is the default for recognition. CPU installations must use `--device cpu`
  or `execution.gpu.device: cpu` in recognition YAML.

Both runtime profiles include tagging, face/body recognition, all supported pose
backends (including RF-DETR-XL and ViTPose++), and the TPE solver. Keep exactly one
ONNX Runtime distribution installed.

| File | Purpose |
| --- | --- |
| `requirements.txt` | Default GPU installation; includes `requirements-gpu.txt`. |
| `requirements-gpu.txt` | Complete runtime with CUDA PyTorch and GPU ONNX Runtime. |
| `requirements-cpu.txt` | Complete runtime with CPU PyTorch and CPU ONNX Runtime. |
| `requirements-dev.txt` | Test tools; add to either CPU or GPU profile. |

No model-specific requirements files are needed. Model downloads and feature
activation still follow the YAML and CLI settings. See the [recognition setup
guide](docs/facial-recognition.md), [development setup](docs/development.md) and
[dependency upgrade instructions](docs/maintenance/updating.md#refresh-python-dependencies).

## Quick start

Run these commands from the repository root in a Python 3.14.7 environment:

```powershell
python -m pip install -r requirements-gpu.txt -c constraints/gpu-py314.txt
# CPU alternative: requirements-cpu.txt with constraints/cpu-py314.txt

$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<your Stash API key>"

python .\stash_ai.py --download-models --models all
python .\stash_ai.py --models all --scene-id 123 --dry-run --verbose
python .\stash_ai.py --models all --scene-id 123
```

Set `STASH_API_KEY` if your Stash instance requires authentication; without it,
the client sends no `ApiKey` header. In Bash, use `export STASH_URL=...` and
`export STASH_API_KEY=...` in place of the PowerShell assignments.

Review [tagger_models.yml](tagger_models.yml) before a full run. Selected models
must explicitly define `scene_threshold`, `marker_threshold` and `batch_size`.
Matching cached results and compatible checkpoints are reused automatically.

## Common operations

```powershell
# Tag all eligible scenes, reusing matching evidence.
python .\stash_ai.py --models all

# Re-publish stored evidence after policy changes; no model inference.
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Preview cleanup of excluded results.
python .\stash_ai.py --models all --cleanup-only --dry-run

# Inspect the tagging database without contacting Stash.
python .\stash_ai.py --db-report
```

For recognition with the shipped RF-DETR-XL and ViTPose++ Large configuration:

```powershell
python .\stash_ai.py recognition run --scene-id 123
python .\stash_ai.py recognition portraits --performer-id 456
```

Recognition requires an explicit scene selection (`--scene-id` or `--all-scenes`).
Portrait refresh reuses retained memberships; see [portraits](docs/recognition/portraits.md).

## Configuration and state

| File | Purpose |
| --- | --- |
| [tagger_models.yml](tagger_models.yml) | Tagging models, thresholds, batches and workers |
| [temporal_policy.yml](temporal_policy.yml) | Temporal rules and combined publication |
| [tag_hierarchy.yml](tag_hierarchy.yml) | Shared tag categories |
| [scene_tag_exclusions.yml](scene_tag_exclusions.yml) | Global tag exclusions |
| [marker_core_exclusions.yml](marker_core_exclusions.yml) | Marker primary-tag exclusions |
| [facial_recognition.yml](facial_recognition.yml) | Recognition models, matching, portraits and resources |
| `tagger_sources.sqlite3` | Tagging evidence, checkpoints and publication state |
| `faces_state.sqlite3` | Recognition evidence and identity/publication state |

Back up your databases and local configuration. Use the [database guide](docs/maintenance/database.md)
for upgrades and repairs, and the separate [recognition state guide](docs/recognition/storage.md)
for recognition retention and compaction.

## Documentation

Start with the [documentation index](docs/README.md) or choose a task:

- [Tagging, merge and cleanup](docs/tagging/README.md)
- [Recognition, detection audits and portraits](docs/facial-recognition.md)
- [Merge solver and report interpretation](docs/recognition/solver.md)
- [Updating the application](docs/maintenance/updating.md)
- [Legacy installation migrations](docs/maintenance/legacy-migrations.md)
- [Tests and CI](docs/development.md), [architecture](docs/ARCHITECTURE.md) and [benchmarks](benchmarks/README.md)

Run `python stash_ai.py --help` or `python stash_ai.py recognition --help` for CLI options.

## Security

Keep credentials in the environment and out of the repository. Read-only requests
may be retried; Stash mutations are not automatically retried after an uncertain
response. `--dry-run` does not write to Stash or the persistent source database,
but can download model assets and perform inference. See [security guidance](SECURITY.md).

## License

Original source code is licensed under [GPL-3.0-only](LICENSE). Third-party
components and model weights retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
