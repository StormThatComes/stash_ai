# Evidence index measurements

Measured offline on CPython 3.14.7, using identical synthetic inputs and exact
result comparisons. These are CPU hot paths, not a full-library speed estimate.

| Workload | Previous | Indexed |
|---|---:|---:|
| 4,000 distinct overlapping contributors | 414.7 ms | 8.24 ms |
| 4,000 containment queries over 4,000 disjoint intervals | 297.3 ms | 0.74 ms |

Shared coverage uses binary searches. Contributor deduplication first partitions
by cheap scalar fields, then promotes large collision groups to structural hash
buckets. Exact equality remains authoritative, including hash collisions; proof
sharing and borrowed objects remain intact.

Extras use 64-frame tag blocks for narrow queries and the original tag inventory
for wide queries. Per-source maxima use a lazy tree of 64-observation maxima,
with exact boundary scans. Additional index memory is linear and released with
the scene's ObservationIndex. Overlapping or partially clipped timelines retain
the reference scanner.

The retained-evidence and full SQLite publication benchmarks at 160 frames / 12
labels produced identical logical snapshot and Stash hashes. At this small size,
full publication measured 117 ms before / 135 ms after in short local trials;
index overhead does not imply an improvement for every workload. The large
quadratic cases above are the reason for the change.

Removed scene-cap quality ranking now uses canonical tag/time/group ordering.
Scores remain separate per model; no prediction version or raw evidence changes.
Existing cache can be merged without another inference pass.
