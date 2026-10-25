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
