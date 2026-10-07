# Deferred improvements

The October 2026 improvement stack keeps these items out of its active scope:

| Audit item | Change | Revisit when |
|---|---|---|
| 2 | Executable multi-model commands in the offline cache backfill plan | The generated plan is used regularly or executed automatically. Normal CLI inference already finishes selected models before merge. |
| 8 | Local Stash publication lock shared by different state DB paths | Several state databases are used concurrently against the same Stash instance. The existing per-DB lock and single publication writer remain supported. |
| 24 | Real ONNX Runtime CPU integration and optional GPU provider smoke | ONNX backend/preprocessing changes or broader integration testing. |

All active improvements preserve valid retained predictions. Refactoring,
optimization, reporting and transport changes must not request fresh inference
merely because code or a publication fingerprint changed. Prefer metadata repair,
offline migration and merge replay. When observations are actually missing or
unrecoverable, explain the reason and limit inference to affected scenes/models.
