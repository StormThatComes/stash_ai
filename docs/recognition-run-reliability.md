# Recognition run: face and person evidence reliability

The recognition rebuild now commits face assignments, hybrid recovery, independent
body assignments, identity aliases, and publication intent at one SQLite boundary.
If a later stage fails, the prior complete assignment set remains available. Raw
scan checkpoints are still retained independently for resuming collection.

## Assignment rules

- A measured, admitted face can veto a contradictory body attachment, including
  when that face has no identity membership yet. Missing faces remain unknown.
- Simultaneous hybrid claims are checked together against accepted appearances.
  Body-only similarity cannot justify assigning two simultaneous bodies to one
  identity. Existing measured-face reflection rules remain in place.
- Global body consolidation includes face-only presence and faces linked to body
  observations in its contradiction checks.
- Disabling auxiliary face matching also disables auxiliary hybrid recovery.
  Missing pose keypoints cannot establish face/person ownership at image borders.

## Stable identities and independent support

Body label reuse follows the physical scene/input/cue/geometry, so changing
collection thresholds or scan policy does not by itself create a new identity.
Historical body banks with an older matching policy continue to compete as rivals;
they must be rebuilt before participating in a new merge.

Repeated retained pixels contribute one independent support vote and one body
scoring witness. Equal embeddings from different pixels retain their original
weight; computation can be shared without inventing support.

When global body matching is enabled, provisional scene cores can be confirmed
across scenes. Each side must have at least `identities.attach_min_support`
independent witnesses. Their combined independent support must satisfy both local
and global minimums, span at least two scans, and pass the ordinary similarity,
cohesion, ambiguity, face contradiction and co-presence gates. The complete set of
involved scenes must be selected. Original witnesses used to establish this first
confirmation are retained explicitly; ordinary later attachments do not add votes.

Face landmark ordering is evaluated in the eye coordinate frame, including for
upside-down faces. The scan quality format changes to invalidate old decisions;
the next run recollects scans under the corrected quality contract. Physical
origins preserve identity labels across that recollection.

## Resource use and recovery

Complete hybrid mode witnesses are spooled to temporary disk storage. Selection,
mode checks and score reductions stream bounded working sets. Full witness banks
are not serialized to parallel workers. Temporary files are released with their
catalogue, including on failed construction. Sufficient temporary disk space is
required for retained witnesses.

Local body matching reuses exact dot products and weighted banks for repeated
vectors. Configured comparison limits still apply to distinct comparisons; highly
diverse worst-case inputs can still exhaust them and fail without a partial
rebuild. No approximate nearest-neighbour acceptance or reference truncation is
introduced by this optimization. Warm body results require unchanged configuration
and global evidence versions, including unselected rivals and identity aliases.

Person inference batches can span cues while face/body data remain in the same
checkpoint transaction. A failed person model is closed and replaced before a
later scene uses it. Enabling crop retention or increasing the crop budget can
backfill missing crops from retained geometry and source frames without loading
models or changing embeddings.

Run history and progress distinguish rejected faces, rejected independent body
observations, and unavailable face-associated person evidence, with reason counts.
The existing `observations_rejected` counter continues to count face rejections.
