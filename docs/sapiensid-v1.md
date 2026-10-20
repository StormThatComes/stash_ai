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
  112x112 face), **19 keypoints**, and body/face geometry. Optional
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
