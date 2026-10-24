# Automatic recognition follow-up series

VTT inputs only. Keep one GPU owner. Manual identity corrections and a new
hand-labelled corpus are deferred. Tuning and report-merge simulation remain
in scope; unlabeled trials cannot certify false-merge accuracy.

1. Profile/catalog: stream historical representatives, constant-time matrix
   accounting, SQL fetch timing and coordinator pipeline timing. Worker-side
   queue/copy measurements will accompany the resource and VTT changes below.
2. Representatives: alignment measurements, visibility-aware quality and stable
   per-scene selection with full refresh of changed identities.
3. Matching: supported appearance scoring, independent evidence units and
   indexed query witnesses. Preserve rival and co-presence vetoes.
4. Compute: provably safe screening, shared immutable process payloads and a
   common resource budget; extend worker telemetry.
5. VTT: cross-cue microbatches and per-sprite decode coordination. Incorporate
   the existing VTT pipeline PR before changing these paths.
6. Replay: dependency-driven retries on frozen generations and late checkpoints.
7. Tuning: isolated replay search and report-merge impact simulation; optional
   labels, proposed YAML export, no automatic source configuration changes.

Each PR builds on its predecessor. Verify numerical/decision equivalence for
performance-only changes; exercise threshold boundaries, false-merge scenarios,
rollback, cancellation and resumption for behavioral changes. Real GPU speed and
recognition accuracy require representative hardware/data and are not inferred
from synthetic tests.
