# Development and validation

Use Python **3.14.7**, the version pinned by the [CI workflow](../.github/workflows/tests.yml).
From the repository root, install the CPU application and development tools:

```powershell
uv pip install -r requirements-cpu.txt -r requirements-dev.txt -c constraints/cpu-py314.txt --index-strategy unsafe-best-match
python -m pytest -q
```

`requirements-dev.txt` is a tools-only overlay: replace the CPU profile and
constraints with their GPU counterparts to develop in a CUDA environment.
The runtime profiles already include Optuna and all inference dependencies.

Most tests use mocked Stash and model backends. Optional real-model smoke tests
need their dependencies/assets and explicit environment flags; a production Stash server is not required.

## CI coverage

The workflow runs the main suite in three Linux shards, updater tests on Linux
and Windows, CPU/GPU dependency resolution for both platforms, and recognition
CPU/model and Windows lifecycle/video jobs. Real-model smoke jobs download pinned
assets. Hosted CPU coverage does not establish CUDA performance or recognition accuracy.

See [benchmarks](../benchmarks/README.md) for reproducible measurement commands and
[recognition evaluation](recognition/evaluation.md) for accuracy and retained-evidence pilots.
Historical test counts belong in the [archive](archive/README.md).

## Documentation ownership

- Keep the root README focused on installation and first commands.
- Add current workflows to the relevant tagging, recognition or maintenance guide
+  and link them from the [documentation index](README.md).
- Link to the canonical topic instead of repeating parameter tables or long explanations.
- Keep dated measurements and completed PR plans in `docs/archive/` or `docs/validation/`.
- Check relative links, section anchors and example configuration after moving content.
+  Commands are run from the repository root, regardless of the guide's location.

The [architecture reference](ARCHITECTURE.md) describes code ownership and module boundaries.
