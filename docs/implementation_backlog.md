# Deferred improvements

The October 2026 improvement stack keeps these items out of its active scope:

| Backlog | Change | Revisit when |
|---|---|---|
| B1 | Executable multi-model commands in the offline cache backfill plan | The plan is executed regularly. Finish selected models before final merge, without publication between models. |
| B2 | Local Stash publication lock shared by different state DB paths | Several state databases are used concurrently against the same Stash instance. |
| B3 | Real ONNX Runtime CPU integration and optional CUDA provider smoke | ONNX backend/preprocessing changes or hardware integration testing. |
| B4 | Streaming and limits for detailed reuse-report/history exports | Large exports become a measured memory or latency problem. |
| B5 | Full core test suite on Windows | Expand beyond the existing Windows updater checks. |
| B6 | Shared preparation for model utility/ablation analyses | A separate benchmark demonstrates a gain while preserving qualification and consensus. |
| B7 | Pin GitHub Actions to full commit SHAs with controlled updates | A separate CI dependency maintenance round. |
| B8 | Explicit package sources/hashes for CPU/GPU installations on Linux/Windows | A separate reproducible-installation round. |
| B9 (latest top30 #4) | Byte budgets and admission for input queues, tensors/batches and overlapping scenes | Design and measure the shared budget across two scenes. Explicitly excluded from the current eight PRs. |
| B10 (latest top30 #28) | Standalone retention stress benchmark with realistic vocabularies, long scenes and extreme buffers | Measure RSS and evidence equality in a separate performance round. Explicitly excluded from the current eight PRs; ordinary regression tests remain in scope. |
| B11 ([#312](https://github.com/StormThatComes/stash_ai/issues/312)) | End-to-end wall-clock deadline for updater GitHub downloads | Revisit when updater stalls or a separate HTTP deadline-hardening round is prioritized; user-deferred on 2026-10-11. |
| B12 ([#336](https://github.com/StormThatComes/stash_ai/issues/336)) | Test and harden redaction of encoded API tokens in diagnostics | Revisit during a dedicated transport/logging security review; user-deferred on 2026-10-11. |

All active improvements preserve valid retained predictions. Refactoring,
optimization, reporting and transport changes must not request fresh inference
merely because code or a publication fingerprint changed. Prefer metadata repair,
offline migration and merge replay. When observations are actually missing or
unrecoverable, explain the reason and limit inference to affected scenes/models.
