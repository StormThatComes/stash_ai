# Reproducible Python 3.14.7 environments

The CPU and CUDA 12.6/cuDNN 9 variants pin the complete runtime and development
resolution, with exactly one ONNX Runtime distribution. They target CPython
3.14.7 on Linux x86_64 and Windows amd64; CI resolves binary wheels for both.
The GPU job checks installation compatibility, without claiming GPU execution.

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
