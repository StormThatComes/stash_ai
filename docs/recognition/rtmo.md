# Experimental RTMO-S backend

RTMO-S body7 predicts person boxes and COCO17 keypoints in one full-frame ONNX
forward. Selecting it does not load YOLO, RF-DETR, D-FINE or a top-down pose model.
The official 640×640 export, file size and SHA256 are pinned in
`stash_ai/recognition/rtmo_registry.py`; downloads retain atomic checksum verification.
CPU and CUDA use the application's existing ONNX Runtime provider policy.

Merge these options into your recognition YAML:

```yaml
pose:
  detector:
    backend: rtmo
    frame_batch_size: 4
  estimator:
    backend: rtmo
    rtmo:
      directory: ''
      torso_nms: false
  deduplication:
    enabled: false
  refinement:
    enabled: true
    generic_second_pass: false
    bidirectional_prompts: false
```

Both backend selectors must agree. Tune Detector switches the paired selector
when choosing RTMO and uses the same detection path as recognition/audit. Normal
person score/NMS/count and keypoint quality thresholds remain active. Dynamic ONNX
batches use the configured frame batch; a fixed-one graph processes one frame at a
time. Outputs are validated and restored from bottom/right letterboxing to source
coordinates before common face association and body collection.

SAM can generate masks and withhold foreign/disconnected points using RTMO's
measured first-pass keypoints. Confidence is explicitly a proxy, not measured
visibility. RTMO does not support top-down crop retries, generic masked second-pass
pose inference, or bidirectional borrowed-point verification. These modes are
rejected instead of silently pretending to verify points. `torso_nms` optionally
preserves overlapping results with distinct reliable torsos, but cannot recover
candidates removed inside the exported graph.

Changing RTMO thresholds/torso policy changes the dependent cache/evidence contract.
Asset paths do not. Defaults remain unchanged. Use existing `audit-detection` and
Tune Detector side-by-side comparisons on the same cues before selecting this backend
for a full library. Synthetic geometry tests and a pinned-graph CPU smoke test do not
establish contact-scene accuracy or CUDA throughput.

Upstream references:
- https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo
- https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmo/body7/rtmo_body7.md
- https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmo.py
