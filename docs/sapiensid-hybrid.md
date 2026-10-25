# Supplemental SapiensID evidence

The hybrid integration preserves the configured face recognizer and its namespace.
SapiensID features belong to individual observations, not mutable performer IDs.

The first increment introduces schema version 6 and the `sapiensid` configuration
section. Collection, fusion and command integration follow in the stacked PRs;
this increment alone does not enable hybrid recognition. `enabled` defaults to
false. Matching defaults are uncalibrated starting values, not accuracy claims.

Writer initialization migrates schema versions 1–5 transactionally. Offline
reports can still read version 5 without migrating it. Supplemental data records
an input contract, availability status, metadata, a normalized 4096D float32
embedding and optionally a person crop. An unavailable input is not a negative
similarity score. Inference failures must propagate rather than being stored as
unavailable evidence.

Changing matching thresholds or local model paths preserves the vector contract.
Changing person detection/preparation semantics changes that contract. Supplemental
writes invalidate existing scan dependency certificates through the established
observation triggers. Face namespaces and face scan identities remain unchanged.

## Collection increment

The second increment collects supplemental embeddings during `recognition run`,
including when the face scan is reused. It does not yet merge identities using
those embeddings. Matching and execution budgets are reserved for the remaining
increments; collection currently uses scalar inference.

Install `requirements-sapiensid.txt` alongside the normal runtime. Obtain the
upstream wb4m checkpoint and model.yaml from the official SapiensID repository,
and a trusted YOLOv8n-pose checkpoint. The adapter verifies the pinned source
revision and checkpoint hashes, sharing the existing PyTorch installation.

```yaml
sapiensid:
  enabled: true
  model:
    source_directory: ../sapiensid
    checkpoint_directory: recognition_models/sapiensid_v1/wb4m
    model_yaml_sha256: 2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9
  pose:
    path: recognition_models/sapiensid_v1/yolov8n-pose.pt
    sha256: REPLACE_WITH_VERIFIED_LOWERCASE_SHA256
```

Keep `models.recognizer` set to your face recognizer. Source must be a clean
checkout of `mk-minchul/sapiensid` at
`a8d8011117db23f20c7ade3bc38e8075810ac886`. The required model.pth hash is
`8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02`.
Relative paths resolve from the configuration file.

Association checks include rejected face detections to avoid hiding competing
faces. Embeddings are collected for eligible retained face observations only;
body-only recognition and rescuing faces rejected by the scan gate are not
implemented. Missing/ambiguous associations are explicit unavailable records.
Inference failures fail the scene, preserving only previously completed cues.

`audit-alignment` exports the exact normalized-input reconstruction, whitespace
mask (not semantic segmentation), and 22 measured points. It loads person pose
weights but not identity weights. `report-identity` exports retained person PNGs
and metadata offline. Missing person images are recorded, never reconstructed
from a face crop. This adapter uses SCRFD/YuNet face landmarks with YOLO pose;
its accuracy equivalence to the upstream DFA pipeline has not been established.

Upstream SapiensID source/weights use CC BY-NC 4.0. Ultralytics dependencies have
separate licensing. No model weights are included in this repository.
