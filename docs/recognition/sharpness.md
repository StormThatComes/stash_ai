# Foreground sharpness

All face/body admission floors, anchor/query floors and portrait/reference ranking
by default consume one **0–100 directional reblur score**, with larger values indicating
more resolved detail. Motion blur and optical defocus lower the same score; an optional split estimator is described below and requires no additional model. This is an uncalibrated sharpness
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

Recognition measurements use the pixels presented to the identity encoder:
**112x112 aligned face inputs** and **384x384 square-padded SapiensID inputs**.
Face scanning reuses the prepared encoder tensor, including each recognizer's
channel order and normalization. SapiensID RGB and support masks share the exact
float bilinear resize, with no change to its interpolation/antialias settings.
Pixels are measured in ordinary gray intensity units, before normalization or
after reversing it; normalized model channels are not interpreted as gray.

The face visibility mask is projected from FaceOcc coordinates into the aligned
112px input, accounting for rotation, crop placement and pixel-center conventions.
All bilinear mask contributors must be valid. Body masks undergo the same square
padding and resize as the model pixels. Synthetic padding, mask holes and encoder
background-replacement outlines never count as resolved detail. Mask-consumer
switches remain independent; source pixel minimums remain source pixel minimums.

`sharpness` controls recognition admission and anchor/query gates. The separate
`portrait_sharpness` retains the existing portrait measurement (256px FaceOcc for
faces and source crop for bodies), so a good tiny encoder input cannot overwrite
the larger portrait's quality preference. This does not introduce a new portrait
rendering/quality algorithm. Legacy records without the separate field retain
their existing portrait behavior; refreshed observations store both values.

This input-space change requires another threshold calibration and recognition
rerun even if directional reblur scores were already collected. A small sharp
source can lose usable detail after enlargement; a larger mildly blurred source
can become adequate after reduction. Separate face/body thresholds still make
sense. Compare sharp and blurred **real VTT** examples when choosing floors;
synthetic tests verify regressions, not a universal threshold.

## Measurement

1. Apply mild 5x5 binomial smoothing and measure directional derivatives at
   0, 45, 90 and 135 degrees. Images are not resized by this metric.
2. Reblur those derivatives at two scales (5- and 9-tap binomial filters).
3. Compute local derivative variances in 32x32 tiles. Removing the local mean
   avoids interpreting a linear lighting gradient as an informative edge.
4. Estimate noise per tile using the mixed second derivative. Subtract its
   expected contribution using each filter's squared kernel norm. Ignore tiles
   without at least 16 supported samples and sufficient signal above noise.
   Exclude tiles where reblur increases energy by moving a neighbouring edge
   into the tile; that is a boundary effect, not evidence of blur.
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
scalar summaries for visibility and both sharpness scales, not image/mask arrays. Noise suppression always
runs; `quality_diagnostics` only controls retention of its diagnostic value.

Records include `sharpness_algorithm: foreground-directional-reblur-v1`, the
measurement region in `sharpness_method`, `sharpness_pixels` (supported centers),
`sharpness_contrast` and optional `sharpness_noise` (estimated measured-gray sigma).
Encoder measurements also include `sharpness_space: encoder-input-sharpness-v1`
and `sharpness_input_size`. Bodies retain source dimensions and the resize scale.
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

## Optional focus and motion thresholds

At `face.quality`, `body.collection`, `matching.face.samples.minimum`,
`matching.face.anchors.quality`, and `matching.body.anchors`, the following fields
are available (values below are illustrative, not calibrated defaults):

```yaml
sharpness_mode: split
min_sharpness: 10.0        # fallback when the split measurement is unavailable
min_focus_sharpness: 40.0
min_motion_sharpness: 60.0
```

The default is `sharpness_mode: legacy`, with both new floors zero. Legacy scores
and decisions are unchanged. Each stage selects its own mode. On a reliable
measurement, split mode replaces that stage's legacy floor with the two component
floors; both must pass. Other stages still apply their own configured gates. For
example, switching scan admission alone does not disable a legacy anchor floor.
A face profile override applies to the legacy fallback only. Portrait ranking
continues to use the separate portrait-scale score, not these recognition gates.

`focus_sharpness` estimates isotropic softness and `motion_sharpness` estimates
directional smear. Both range from 0 to 100, with higher meaning sharper; neither
is a probability, a percentage of image quality, nor convertible from the legacy
score. Motion and defocus may coexist. The method cannot establish the physical
cause of blur: resizing, complex camera motion, texture and compression can
produce ambiguous or misleading evidence. Real VTT calibration is required.

The optional estimator fits an edge-spread covariance from four supported edge
normal directions. It samples monotonic edge profiles within the same eroded
foreground support, rejects oscillating/ringing profiles, requires local signal
above noise, and pools widths from multiple tiles before fitting isotropic and
directional components. `sharpness_confidence` is a fit-consistency indicator,
not a calibrated probability. Unsupported, noisy, tiny or one-direction patches
produce `focus_sharpness: null` and `motion_sharpness: null`; missing evidence is
never represented as perfect quality. Strong mixed blur may also be unavailable.
The implementation shares NumPy/Torch arithmetic and transfers seven scalar
summaries per image, without transferring device image arrays or loading a model.

Inspect Scene/Performer admission details show the active component gates, their
measurements and fit confidence, or explicitly label the legacy fallback.
Tune Detection exposes the mode and thresholds in face quality/body collection.
In split mode its final-result caption shows both measurements and failed blur
gates. Face previews use an aligned 112px **ellipse mask**, without loading
FaceOcc, and are labeled accordingly; production FaceOcc masking can give a
different result. Body previews use the same encoder-size preparation and mask
policy as recognition. Detection remains unfiltered in the tuner. The preview
runs only when split mode is selected, and is not persisted as recognition data.

Rerun recognition for affected scenes to obtain the new measurements. Scan,
visibility and body quality contracts refresh; older body banks remain frozen
competitors. The merge solver replays the same split/fallback decision using
retained component measurements; a replay cannot derive missing components from
an old scalar score. It does not automatically switch sharpness modes or search the new component
floors; these stay fixed at the configured values during solver exploration.
