# Reproducible Python 3.14.7 environments

The CPU and CUDA 12.6/cuDNN 9 variants pin the complete runtime and development
resolution, with exactly one ONNX Runtime distribution. They target CPython
3.14.7 on Linux x86_64 and Windows amd64; CI resolves both complete profiles.
Compiled packages require wheels; only the pure-Python `iopath` and
`antlr4-python3-runtime` dependencies may build from source.
The GPU job checks installation compatibility, without claiming GPU execution.
Both profiles install the complete application: tagging and on-demand recognition
with YuNet/SFace/AuraFace, all supported body/person/pose backends, and Optuna.
`requirements.txt` selects the GPU profile by default.
There are no separate recognition installation/constraint profiles. Development
tests use the tools-only `requirements-dev.txt` in addition to the selected
application profile. The development overlay does not select a torch or ORT build.

Recognition defaults to `execution.gpu.device: cuda`. For CPU runs explicitly set
`execution.gpu.device: cpu` in `facial_recognition.yml` or use `recognition run --device cpu`.
An existing environment must contain only its selected ORT distribution; remove
the other one before changing profiles. Use only `opencv-python` (required by
Ultralytics), not a second distribution sharing `cv2`; see the
[upgrade instructions](../docs/maintenance/updating.md#refresh-python-dependencies).

```sh
python -m pip install -r requirements-cpu.txt -c constraints/cpu-py314.txt
# Or, in a separate environment with a compatible NVIDIA driver:
python -m pip install -r requirements-gpu.txt -c constraints/gpu-py314.txt
```

Keep the general requirements for users managing their own environment. Refresh
both constraints deliberately and review the diff; dependency maintenance does
not invalidate cached inference by itself. Prediction semantics and input/model
identities continue to determine reuse.

```sh
uv pip compile requirements-cpu.txt requirements-dev.txt --python-version 3.14.7 --universal --index-strategy unsafe-best-match --emit-index-url -o constraints/cpu-py314.txt
uv pip compile requirements-gpu.txt requirements-dev.txt --python-version 3.14.7 --universal --index-strategy unsafe-best-match --emit-index-url -o constraints/gpu-py314.txt
```
