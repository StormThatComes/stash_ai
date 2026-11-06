# Updating Stash AI

[Documentation index](../README.md)

`update_stash_ai.py` updates managed files without Git or `pip`. The updater
requires `GH_TOKEN` (or `GITHUB_TOKEN`) with repository `Contents: read` access.
For a flat-layout installation, first follow the [legacy bridge](legacy-migrations.md#flat-layout-updater-bridge).

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"

python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run
python .\update_stash_ai.py
```

The updater:

- downloads application files from a single `main` commit SHA;
- installs launchers, the `stash_ai/` package, YAML files, all four application/development
  dependency manifests, CPU/GPU constraints,
  the root README and Markdown/text documentation under `docs/`;
- preserves locally modified YAML and writes `.incoming` proposals;
- refuses to overwrite locally changed code, documentation or dependency files without `--force-code`;
- does not update models, `.env`, SQLite or Python dependencies;
- uses an installation lock and recovery journal.

## Refresh Python dependencies

When upgrading from the separate feature profiles, remove both OpenCV
distributions before reinstalling the selected profile:

```powershell
python -m pip uninstall opencv-python opencv-python-headless
```

The complete profiles use only `opencv-python`, required by Ultralytics, so two
packages cannot overwrite the same `cv2` module. Minimal Linux containers may
also need system OpenGL/GLib libraries (`libgl1` and `libglib2.0-0` on Debian/Ubuntu).

After updating, prepare or refresh the local environment manually using the
downloaded dependency files. Choose one application profile:

```powershell
python -m pip install -r requirements-gpu.txt -c constraints/gpu-py314.txt
# Or, for a CPU environment:
python -m pip install -r requirements-cpu.txt -c constraints/cpu-py314.txt
```

`requirements.txt` selects the GPU profile. `requirements-dev.txt` is available
for development and tests. The updater downloads these manifests but does not
run `pip` or modify the installed Python packages.

The shipped recognition YAML enables RF-DETR-XL person boxes and ViTPose++ Large.
Their inference dependencies are included in both CPU/GPU profiles.

For RF-DETR-XL setup and weight terms, see
[model setup and terms](../recognition/pose-backends.md#rf-detr-xl-person-detector).

The CPU/GPU profiles also include SapiensID inference dependencies. With
`body.enabled: true` and
`execution.models.download_missing: true`, the application downloads missing upstream
source and model checkpoints into the SapiensID paths configured in
`facial_recognition.yml` when inference first needs them. The default
`body.encoder.variant: wb12m` selects the model trained on more data;
`wb4m` remains available. See
[the SapiensID guide](../sapiensid-hybrid.md) for custom destinations and offline use.

If the installed updater predates the consolidated profiles, run it once to
update itself, then run it again to retire tracked feature-specific manifests.
Locally modified manifests retain the usual conflict and backup protection.
The second run can add newly supported files even when `main` has not changed.

After an interrupted update:

```powershell
python .\update_stash_ai.py --recover
```

Locally modified YAML is preserved with `.incoming` proposals. Use
`--overwrite-configs` only when intentionally accepting repository configuration.
