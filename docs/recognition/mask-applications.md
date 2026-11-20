# Independent person-mask applications

`pose.mask` generates and validates one instance mask per selected person.
`pose.mask.apply_to` independently selects its consumers. It works with PMPose,
ViTPose++, Sapiens and mask-only RTMO, in recognition and Tune Detection.
FaceOcc and the body encoder's mandatory padding mask are separate mechanisms.

For masked sharpness and encoder input without changing the pose:

```yaml
pose:
  mask:
    enabled: true
    backend: sam_pose2seg
    mode: all
    apply_to:
      sharpness: true
      body_encoder_input: true
      pose_refinement: false
      keypoint_filtering: false
      box_expansion: false
      overlap: false
      body_admission: false
```

These are strict YAML booleans. `mode` still selects `all` people or only
`overlap` cases; the backend, prompts, cleanup and mask-validity thresholds
live directly under `pose.mask`.

| Application | true | false |
| --- | --- | --- |
| `sharpness` | Interior mask sharpness on original source pixels | Full original person-crop sharpness |
| `body_encoder_input` | Neutral background in validated, separated body crops | Original crop pixels |
| `pose_refinement` | Masked second pose pass and dependent contact recovery | No mask-driven pose re-estimation |
| `keypoint_filtering` | Per-joint mask ownership/fusion and ownership fallback | Preserve selected pose predictions before ordinary estimator output filters |
| `box_expansion` | Union of original box and admitted mask extent | Original person box |
| `overlap` | Directional mask overlap with existing missing-mask fallback | Box overlap, ignoring own and peer masks |
| `body_admission` | Mask separation/conflicts may admit query-only bodies or reject observations | No admission/crowd-context decisions based on those mask flags |

`sharpness: false` changes the measurement region; it does not disable the
threshold. `body.collection.min_sharpness` and `matching.body.anchors.min_sharpness`
continue to apply. Recalibrate their values when changing measurement regions.
Encoder background replacement never enters the sharpness measurement.

A missing/invalid mask with masked sharpness selected has no valid sharpness
measurement and fails a positive sharpness floor. There is no silent full-crop
fallback. Positive masked sharpness floors require enabled masks and `mode: all`,
but do not require a masked second pose pass. With full-crop sharpness selected,
mask generation can be disabled entirely.

Encoder masking retains the conservative separated-person requirement of the
previous `masked_identity_crop` option. Missing/ineligible masks use the full
crop; `embedding_foreground` and `embedding_mask_reason` report that choice.
Pose refinement falls back to the preceding pose when its mask or second pass
cannot be admitted. Disabling `body_admission` never makes an invalid mask valid
for another consumer.

For unchanged pose geometry, disable `pose_refinement`, `keypoint_filtering`
and `box_expansion`. Existing estimator score/presence/visibility filters and
ordinary deduplication still apply. Expanding boxes or changing a pose can
indirectly change face/body association and encoder inputs. There is no direct
mask-pixel face-association switch.

RTMO supports mask consumers but cannot run a masked second pose pass. Enabling
`pose_refinement` for RTMO is an error. Bidirectional borrowed prompts still
require an independent second-pass confirmation.

## Migration and diagnostics

Old v1 and v2 `pose.refinement` configurations remain readable. Exports and
`migrate-config` produce `pose.mask` with explicit booleans. Other existing
refinement parameters keep their names under the new group.

| Previous public field | Canonical field |
| --- | --- |
| `pose.refinement.masked_identity_crop` | `pose.mask.apply_to.body_encoder_input` |
| `pose.refinement.expand_boxes` | `pose.mask.apply_to.box_expansion` |
| `pose.refinement.generic_second_pass` | `pose.mask.apply_to.pose_refinement` for generic estimators |

Legacy PMPose always selected the second pass when refinement was enabled;
migration therefore exports `pose_refinement: true` for PMPose even when the
old generic-only switch was false. Existing mask sharpness, filtering, overlap
and body-admission behavior migrates to true. Specifying both an old field and
its new equivalent is rejected instead of selecting one by YAML order.

The bundled recognition and solver YAML files retain their prior choices.
To use the example above, change their `apply_to` values explicitly.

Tune Detection exposes `pose.mask` and `pose.mask.apply_to`. Phase notes and
retained body metadata record requested mask consumers and applied stages.
Masks can be retained for Inspect Scene even with full-crop sharpness selected.
Changed consumer settings invalidate affected body evidence; rerun recognition
for affected scenes before replaying identity/solver results. Raw model caches
remain reusable where their numerical inputs are unchanged.
