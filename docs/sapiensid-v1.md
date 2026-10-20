# SapiensID v1 integration groundwork (not yet an active recognizer)

This PR deliberately **does not** add `models.recognizer: sapiensid_v1` or
publish SapiensID identities. Selecting an unsupported recognizer still fails
configuration validation. SapiensID is **person re-identification**, not a
drop-in face embedder. Claiming a working backend without body detection,
upstream keypoint estimation, and model-loading integration would produce
incorrect person-to-scene assignments.

## Verified upstream contract

- Official source: https://github.com/mk-minchul/sapiensid
- Weights: the official v1 repo links to a Google Drive archive under
  `tasks/sapiensID/README.md`; the released `wb4m` checkpoint is
  `model.pth` with `model.yaml`. Do not confuse with Meta Sapiens pose.
- License: CC BY-NC 4.0. Validate the intended use before deployment.
- Image input: RGB, square white padding, torchvision float-tensor bilinear
  resize to **384x384**, then mean/std normalization `0.5 / 0.5`.
  `stash_ai.recognition.sapiensid_v1.preprocess` implements this contract.
- Output: **4096-dimensional** feature per person observation; downstream
  L2 normalization and cosine similarity must be calibrated per model.
- Upstream `yolo_dfa.yaml` aligner uses YOLO pose plus DFA face keypoints.
  The model needs a crop for each detected person (not an ArcFace-aligned
  112x112 face), **22 raw keypoints** (17 YOLO pose + 5 face), which the upstream
  `KP19Preprocessor` reduces to **19 model keypoints**, plus body/face
  geometry. Optional
  foreground masking/dynamic patching is model-specific; do not fabricate
  landmarks or pass five face points as the expected body landmarks.
- Confirm the upstream checkpoint's provenance before loading executable
  pickled weights. `verify_wb4m_checkpoint` supports strict local SHA256 and
  size checks, but *does not* load checkpoints, download assets, or make
  PyTorch pickle loading safe by itself.

Reference `wb4m` SHA256 and byte count in
`stash_ai/recognition/sapiensid_v1.py` are published independently at
https://huggingface.co/peabody124/sapiensid-eqx . That HF artifact is an
**Equinox/JAX conversion**, NOT a ready-made ONNX/PyTorch inference model.
We do not require or install JAX.

## Changes in this PR

- Accept bounded `float32` **4096D** vectors in retained-embedding scalar and
  warm-rebuild validators, preserving the 4097+ dimensional rejection.
- Evaluate 4096D vector norms in float64 consistently (including the warm
  native validator): previously the float32 rounding guard alone exceeded
  the 0.001 admissibility tolerance at 4096D, disabling its fast path.
- Add source-compatible SapiensID v1 image preprocessing and strict offline
  checkpoint verification, both tested without downloaded weights.
- No DB schema migration and no change to existing face recognition
  namespace, matching thresholds, existing identities, or Stash assignments.

## Next PR required for real inference / publication

1. Add a verified, independently selectable person detector and pose/keypoint
   adapter. Multi-person VTT frames must produce one observation per **person**,
   including partially visible upper bodies.
2. Provide a pinned, tested v1 model loader (and resource-safe GPU execution)
   using an approved local/Hugging Face checkpoint. Do not silently install
   upstream project's older pinned dependencies into stash_ai's Python runtime.
   Include a trusted loading boundary for `model.pth`.
3. Add a distinct *person geometry* data contract and SapiensID namespace;
   change scan/replay/portrait paths currently hardcoded for 15-value face
   geometry. Retain old face observations rather than converting them.
4. Implement per-person frame association, VTT-coordinate mapping, quality
   gates, and an explicit no-publish pilot for labeled evaluation.
5. Calibrate identity/cohesion/ambiguity thresholds using same/different
   person pairs, **including changes of clothing**. Never reuse AuraFace
   thresholds; test fragmentation *and* false cross-person merges.
6. Measure memory/compute scaling. 4096D is 16 KiB per stored float32 feature,
   versus 2 KiB for a 512D face feature. `clustering.max_reference_bytes`
   remains enforced (128 MiB default). Raise it explicitly for larger
   experiments; do not disable the budget.

**Safety:** Until those steps are complete, the SapiensID code is not a
recognition backend. This is an independently mergeable compatibility PR.

## Next stacked PR: reusable PyTorch inference boundary

`stash_ai.recognition.sapiensid_inference.SapiensIDV1Encoder` now accepts
**already constructed** upstream-compatible PyTorch `model` and `aligner`
objects. The bridge runs CPU or GPU inference synchronously on its owning
thread, preserving the existing 384x384 preprocessing and the official
`aligner(batch) -> (keypoints, masks)` and
`model(batch, foreground_masks=masks, ldmks=keypoints)` contracts.
Outputs are required to be finite 4096D vectors and normalized independently.

The official `KeypointPredictor` returns **22 raw keypoints**: 17 YOLO
body/face pose keypoints and five independently detected face landmarks.
The model's `KP19Preprocessor` then selects **19** of these and derives
body/face regions. Do not pass only five ArcFace/SCRFD face landmarks or
pre-reduce the 22-vector externally. Foreground mask shape is
`N x 384 x 384` with boolean values.

A separate `load_verified_weights(model, path)` uses
`torch.load(weights_only=True)` on the **same file descriptor** that
passed strict size+SHA256 verification. It only accepts tensor state
dictionaries and uses `strict=True` for loading; it does not use the
upstream permissive `strict=False` method or execute arbitrary pickle
classes. Checkpoint/source provenance remains the operator's responsibility.

Example **after** creating official compatible modules in a controlled
environment and obtaining approved weights:

```python
from stash_ai.recognition.sapiensid_inference import (
    SapiensIDV1Encoder, load_verified_weights,
)
# Construct upstream_model and upstream_aligner separately from a trusted
# upstream checkout using the matching model.yaml and yolo_dfa.yaml.
load_verified_weights(upstream_model, "/path/to/model.pth")
encoder = SapiensIDV1Encoder(upstream_model, upstream_aligner, device="cuda")
try:
    feature = encoder.embed(one_person_rgb_pil_crop)  # numpy float32 (4096,)
finally:
    encoder.close()
```

The bridge intentionally does **not** import the upstream project,
register a YAML recognizer, download YOLO/DFA/checkpoints, or modify Stash.
The remaining implementation work is to bootstrap matching pinned upstream
model + aligner classes (their legacy dependencies are not in the JoyTag
profile), provide a real person detector and multi-person crop workflow,
and connect the person-specific geometry and evidence lifecycle before
introducing safe publication. A synthetic CPU adapter test is not a real
recognition-accuracy or GPU-memory benchmark.


## PR 3: Offline initialization from a pinned upstream checkout

The optional bootstrap in `stash_ai.recognition.sapiensid_bootstrap` now
constructs the upstream `SapiensModel` and `KeypointPredictor` automatically,
provided their compatible **local** dependencies and weights are already
installed. It builds the official YOLO-pose + DFA aligner, without invoking
upstream's unsafe `get_face_kp_predictor(...)` loader:

- The SapiensID Python source must be the exact clean revision
  `a8d8011117db23f20c7ade3bc38e8075810ac886` from
  [mk-minchul/sapiensid](https://github.com/mk-minchul/sapiensid).
  There is no network fetch, dynamic GitHub code download, or auto-install.
- Local `model.pth` and `model.yaml` must match the `wb4m` architecture;
  model checkpoint hash+length are checked before loading through
  `torch.load(weights_only=True)` and strict state-dict loading. The model
  YAML and separate YOLO-pose/DFA checkpoint files require operator-supplied,
  independent SHA256 hashes.
- The YOLO pose file is copied to a private temporary verified path and loaded
  by the official Ultralytics constructor. **Ultralytics may use pickle-based
  deserialization internally:** verify upstream trust and the pinned checksum
  before invoking this opt-in bootstrap. Never accept arbitrary user-supplied
  `.pt` files.
- DFA is assembled with upstream's `get_landmark_predictor`, then its
  weights are read from the verified file descriptor using
  `torch.load(weights_only=True)` and loaded strictly. This avoids the
  ordinary upstream `get_face_kp_predictor` that calls unsafe `torch.load`.
- The `KeypointPredictor` is first instantiated with
  `use_yolo=False, face_detector=None` to prevent implicit downloads and
  unsafe weight loads, and then receives the validated YOLO and DFA objects.
- The returned encoder is bound to the calling thread. The existing JoyTag
  torch and torchvision wheels are reused. Extra upstream packages (for
  example `ultralytics`, `omegaconf`, `pyrootutils`, `timm` and their
  dependencies) are **optional**, and must be installed explicitly in a
  compatible environment. Upstream pins are old; do not install their entire
  legacy `requirements.txt` over the main application blindly.

**Offline smoke test on one previously isolated person crop:**

```powershell
python -m stash_ai.recognition.sapiensid_bootstrap `
  --checkout C:\\models\\sapiensid-source `
  --checkpoint-dir C:\\models\\sapiensid-wb4m `
  --model-yaml-sha256 <64_hex_checksum> `
  --yolo-pose-path C:\\models\\yolov8n-pose.pt `
  --yolo-pose-sha256 <64_hex_checksum> `
  --dfa-path C:\\models\\mobilenetv4_Final.pth `
  --dfa-sha256 <64_hex_checksum> `
  --image C:\\models\\one-person-example.jpg --device cuda
```

Hashes for the independently sourced weights can be computed with
`Get-FileHash -Algorithm SHA256 <file>`. Make sure the sources are
trustworthy **before** hashing. The successful smoke test prints only vector
dimensionality, normalized status and execution device; never the biometric
embedding itself. It does not detect multiple people and **does not** change
recognition YAML, SQLite or any Stash scene assignments.

Real-checkpoint/GPU smoke tests are still needed on the intended platform.
The CI tests use synthetic local checkpoints and mocked upstream constructors:
they verify bootstrap lifecycle and rejection behavior, not recognition
accuracy. The next stage is multi-person VTT cropping and a separate
person-observation namespace with a no-publish evaluation.
