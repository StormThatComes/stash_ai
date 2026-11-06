# Face recognizers and runtime

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [SCRFD five-landmark detector choices](#scrfd-five-landmark-detector-choices)
- [Alternative ONNX face recognizers (research weights)](#alternative-onnx-face-recognizers-research-weights)
- [Optional AuraFace recognizer](#optional-auraface-recognizer)
- [Device policy](#device-policy)
- [Shared CPU/GPU runtime and cache compatibility](#shared-cpugpu-runtime-and-cache-compatibility)

## SCRFD five-landmark detector choices

The default detector for new configurations is **SCRFD-10G KPS**. This release
also supports SCRFD-2.5G KPS and the explicitly selected legacy YuNet.
Switching detectors changes the recognition namespace, prevents reuse of
old scan/embedding evidence for that detector, and triggers a new scan;
it does **not** delete historical embeddings or identities.

```yaml
version: 2
face:
  detector:
    backend: scrfd_10g  # default; alternatives: scrfd_2_5g, yunet
  recognizer:
    backend: lvface_b  # or adaface_ir50, auraface, sface
execution:
  gpu:
    device: cuda
```

SCRFD runs in ONNX Runtime on CPU or CUDA with a 640 × 640 letterboxed,
BGR-to-RGB detector input normalized by `(pixel-127.5)/128`. The
nine-output ONNX models use strides 8/16/32, two anchors per location,
five landmarks and per-image NMS. Bounding boxes and landmarks are
converted back into original VTT coordinates; recognizers still receive
the unchanged 112 × 112 ArcFace aligned input. `10G` and `2.5G` are
compute tiers, **not** gigabytes of downloaded data.

The SCRFD artifacts are pinned to Hugging Face commit
`d168e9d89c1687119307f528cbc67eb257cf0021`; 10G weights are
16,923,827 bytes, and 2.5G weights are 3,291,737 bytes. Both have strict
size/SHA-256 integrity checks; only the selected weights are downloaded.

**Licensing:** the HF wrapper advertises Apache-2.0, whereas upstream
InsightFace pretrained model weights have noncommercial research-use
restrictions. Confirm licensing for your intended usage before a
commercial deployment; wrapper metadata does not supersede upstream terms.

The `recognition audit-alignment` command now uses the detector named in
`face.detector.backend`. To compare two variants, run the audit with separate
YAML configurations and **different new output directories**. Compare
actual landmark positions and 112px aligned crops on same-person and
different-person cues, not only face detection benchmark scores.

## Alternative ONNX face recognizers (research weights)

The `face.recognizer.backend` setting supports `sface`, `auraface`,
`lvface_b` (ByteDance LVFace-B Glint360K, 512D, ~456 MB) and
`adaface_ir50` (AdaFace IR50 MS1MV2, 512D, ~174 MB). All use the configured
five-landmark face detector and ONNX Runtime with the existing `execution.gpu.device`
CPU/CUDA policy. No extra PyTorch/InsightFace inference dependencies.

```yaml
version: 2
face:
  detector:
    backend: yunet
  recognizer:
    backend: lvface_b  # or adaface_ir50
execution:
  gpu:
    device: cuda
  models:
    download_missing: true
```

These models are fetched **only when explicitly selected**, from immutable
Hugging Face revisions and checked by SHA-256 plus exact byte count. Both
use an ArcFace 112×112 five-landmark similarity-aligned crop:
LVFace-B uses **RGB** and AdaFace IR50 uses **BGR**; each normalizes
`(pixel - 127.5) / 127.5` and returns L2-normalized 512D embeddings.
The separate embedding namespace prevents mixing any embeddings or identity
evidence between the models, even when both produce 512D vectors. Switching
models triggers a new scan; cached embeddings cannot be converted.

**Weights licensing matters:** LVFace code is MIT, but the upstream model
README explicitly restricts its **pretrained weights** to *non-commercial
research*. The independent AdaFace ONNX exporter notes that its MS1MV2
checkpoint carries dataset/research-use restrictions distinct from the
MIT-licensed AdaFace code. Neither source grants unambiguous permission
for commercial production. Check permissions before using either outside
research, and do not redistribute weights in this repository.

**Calibration:** the example recognition/clustering cosine thresholds
are *not* validated for LVFace or AdaFace. Compare labeled same-person
and different-person samples with the existing internal offline pilot
and `recognition report` diagnostics, then tune independently. Do not carry AuraFace thresholds over as a
measured recommendation. Larger weights and new recognizers do not
automatically recover detail lost in small VTT thumbnails.

Source: https://huggingface.co/bytedance-research/LVFace
and https://huggingface.co/globalnebula/adaface-ir50-ms1mv2-onnx

## Optional AuraFace recognizer

SFace remains the default. Both CPU/GPU application profiles already include
AuraFace's runtime: set `face.recognizer.backend: auraface` in `facial_recognition.yml`
and invoke the same `recognition run --scene-id ID`; no additional requirements
file or ORT package is installed. Exactly one ORT variant is required. CPU is
runtime-tested on Python 3.14.7 on Linux/Windows. The default CUDA device is strict;
CUDA hardware execution is not covered by hosted CPU CI. Actual detector and
embedding providers/runtime/preprocessing provenance are stored in SQLite.

The recognizer uses the pinned publisher's `glintr100.onnx` weight alongside
the selected face detector; no InsightFace dependency, default model pack, landmarks, age/gender
weights or external recognition service is used. The pinned model card and
`LICENSE.md` declare Apache-2.0 for AuraFace; library licenses are separate.

The upstream [ArcFace adapter](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py)
and [five-point alignment](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py)
define a least-squares similarity transform to the 112px ArcFace reference,
BGR-to-RGB conversion, `(pixel-127.5)/127.5` float32 NCHW and normalized embeddings.
The pinned AuraFace graph starts with Conv/PRelu (no built-in initial Sub/Mul),
accepts `data` with dynamic batch and `3x112x112`, and returns `1333` with `1x512`.
Size/hash, runtime I/O, known affine alignment and exact RGB normalization are
checked. This alignment/preprocessing has its own namespace; the SFace namespace
and existing evidence are unchanged.

Switching models computes selected-scene observations in the new namespace;
vectors and ownership bindings are never compared/merged across recognizers.
If this Stash instance already has published identities in another namespace,
new unbound identities are withheld with `incompatible_published_namespace`,
while retaining their analysis. Existing compatible bindings can still be reused.
There is no automatic migration, duplicate-profile creation or hidden review
command. Use the developer pilot for comparison before choosing a model for a new
installation; future cross-model binding/migration is outside this release.

## Device policy

Recognition defaults to `execution.gpu.device: cuda`, including when default YAML is absent.
Both YuNet detection and SFace/AuraFace neural recognition use the application's
ONNX Runtime CUDA provider. CPU image decoding, alignment/NMS and SQLite/identity
matching remain local host work. CPU runs explicitly set `execution.gpu.device: cpu` or
use `recognition run --scene-id ID --device cpu`. Only an explicit `auto` permits
CPU fallback; the CUDA default fails clearly if unavailable or if a session
initializes only CPU. CUDA libraries are preloaded through the same policy as
ONNX taggers, and actual detector/embedding devices are recorded in face SQLite. Weights are checked against immutable revision-bound SHA256 values;
existing custom/corrupt files are never silently overwritten.

## Shared CPU/GPU runtime and cache compatibility

Pinned files, normalized vector spaces and the SFace/AuraFace namespace hashes
remain unchanged when selecting a device. CUDA uses the original OpenCV SFace
alignCrop and raw RGB float32 pixels; normalization remains inside the SFace graph.
YuNet follows OpenCV's BGR/divisor-32 padding, stride decoding, integer-box NMS and
bounded top-K contract. ONNX input/output declarations are adapted in memory for
YuNet spatial dimensions, and legacy initializer-inputs are frozen to their
original values. No neural nodes or initializer values are changed, no transformed
weights are stored, and original assets are checksum-verified first. Transformation
digests and active providers are recorded as runtime provenance. Native CPU ORT
smoke checks validate non-square YuNet inputs and SFace embedding agreement with
OpenCV; CPU/GPU floating-point differences are expected. Device selection alone
does not request a new namespace, model switch or duplicate-profile publication.
