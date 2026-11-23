# Foreground sharpness

All face/body admission floors, anchor/query floors and portrait/reference ranking
now consume one **0–100 directional reblur score**, with larger values indicating
more resolved detail. Motion blur and optical defocus lower the same score; there
is no blur-type classifier or additional model. This is an uncalibrated sharpness
measure, not a probability or an image-quality percentage.

## Updating an existing installation

The YAML names and numeric settings are preserved. **Recalibrate every active
sharpness threshold**; there is no conversion from Laplacian variance to the new
score. Legacy thresholds above 100 remain readable but reject every measured
sample. `face.quality.sharpness_metric: raw` and `normalized` are compatibility
aliases for the same score, including when `profile_min_sharpness` is set. The
metadata field `sharpness_normalized` is also an alias, not a second metric.

Run recognition for the affected scenes and refresh Inspect Scene. Face scan,
FaceOcc quality, body collection/projection and portrait policy versions change,
including when masks or all sharpness floors are disabled. Old body banks remain
recognizable as frozen competitors, not current evidence. Inference pixel caches
that do not contain quality measurements remain reusable. The merge solver must
use these fresh measurements; replay alone cannot recalculate sharpness from old
scalar scores.

Tune these settings separately for faces and bodies:

- `face.quality.min_sharpness` and optional `profile_min_sharpness`;
- `matching.face.samples.minimum.min_sharpness`;
- `matching.face.anchors.quality.min_sharpness`;
- `body.collection.min_sharpness` and `matching.body.anchors.min_sharpness`;
- `portraits.quality.min_sharpness` (a ranking preference, not admission).

Face measurements still use the 256px upright FaceOcc input; bodies still use
source-resolution crops. The algorithm and output units match, but their
different pixel scales and content still justify separate thresholds. Compare
sharp and blurred **real VTT** examples when choosing floors. Synthetic tests
verify regressions, not a universal threshold.

## Measurement

1. Apply mild 5x5 binomial smoothing and measure directional derivatives at
   0, 45, 90 and 135 degrees. Images are not resized by this metric.
2. Reblur those derivatives at two scales (5- and 9-tap binomial filters).
3. Compute local derivative variances in 32x32 tiles. Removing the local mean
   avoids interpreting a linear lighting gradient as an informative edge.
4. Estimate noise per tile using the mixed second derivative. Subtract its
   expected contribution using each filter's squared kernel norm. Ignore tiles
   without at least 16 supported samples and sufficient signal above noise.
5. Measure fractional energy loss under reblur, average the two scales, and
   use the lower quartile of informative tiles for each direction. The weakest
   informative direction determines the final score. An absent direction does
   not penalize naturally directional structure; a sharp patch cannot dominate
   through contrast or pixel count alone.

The complete 15x15 filter footprint must be inside the mask and real source
pixels. No zero-filled mask boundary, padding, masked hole or external texture
enters a valid sample. Small/thin regions with no full footprint have zero
supported pixels; flat or noise-only regions have no positive sharpness evidence.
The existing missing-mask policy and `pose.mask.apply_to.sharpness` switch remain.

NumPy and Torch use the same arithmetic. The FaceOcc device path transfers only
four scalar summaries per image, not image/mask arrays. Noise suppression always
runs; `quality_diagnostics` only controls retention of its diagnostic value.

Records include `sharpness_algorithm: foreground-directional-reblur-v1`, the
measurement region in `sharpness_method`, `sharpness_pixels` (supported centers),
`sharpness_contrast` and optional `sharpness_noise` (estimated source-gray sigma).
Face `crop_sharpness` remains the old rectangular Laplacian diagnostic and must
not be compared with the new thresholds. Body `crop_sharpness`, when a required
mask is missing, is the new full-crop diagnostic; it cannot admit that body.

This method does not guarantee correct ordering for every pair of unrelated
images. Strong compression/ringing, unusual textures, mixed blur and very small
visible regions can remain ambiguous. No sharpness metric can recover detail
that the crop does not contain.

The reblur principle is described by Crété et al., *The blur effect: perception
and estimation with a new no-reference perceptual blur metric* (2007),
[doi:10.1117/12.702790](https://doi.org/10.1117/12.702790). This implementation
uses local noise-corrected gradient energies and conservative directional pooling;
it is not the paper's original metric or a call to scikit-image.
