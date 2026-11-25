"""On-demand comparison of a retained face score with FaceOcc on its verified cue."""
from contextlib import closing
from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw

from .inspection_images import InspectorImages
from .inspection_runtime import Worker, check_cancelled, local
from .models import checked_detection, ensure_assets
from .config import DEFAULT_FACEOCC_LOGIT_THRESHOLD
from .face_region import outline as face_outline, fit as fit_face_region
from .visibility import (FaceVisibility, VisibilityGeometryError, cached, ellipse_radii, foreground_mask,
                         prepare_measurement, valid_fraction)


def ellipse_roi(center, horizontal, radii, source_size, affine):
    """Project an optional comparison ellipse onto the measured crop."""
    center = np.asarray(center)
    c, s = horizontal
    axes = np.array([[c, -s], [s, c]])
    reverse = np.linalg.inv([affine[:3], affine[3:], (0,0,1)])
    mapped = (reverse @ [*center, 1])[:2]
    basis = reverse[:2,:2] @ axes @ np.diag(radii)
    extents = np.linalg.norm(basis, axis=1)
    if (mapped-extents < 0).any() or (mapped+extents > 256).any():
        return None  # Never compare using a clipped denominator.
    yy, xx = np.ogrid[:256,:256]
    sx = affine[0]*(xx+.5) + affine[1]*(yy+.5) + affine[2]
    sy = affine[3]*(xx+.5) + affine[4]*(yy+.5) + affine[5]
    dx, dy = sx-center[0], sy-center[1]
    region = ((c*dx+s*dy)/radii[0])**2 + ((-s*dx+c*dy)/radii[1])**2 <= 1
    total = int(region.sum())
    region &= (sx >= 0) & (sx < source_size[0]) & (sy >= 0) & (sy < source_size[1])
    angles = np.linspace(0,2*np.pi,129)
    points = np.stack((radii[0]*np.cos(angles),radii[1]*np.sin(angles)),axis=1) @ axes.T + center
    return region, total, [tuple(p) for p in points]


def legacy_roi(face, source_size, affine):
    x, y, width, height = face.box
    return ellipse_roi((x+width/2,y+height/2),(1,0),ellipse_radii(width,height),source_size,affine)


@dataclass
class FaceReplay:
    image: Image.Image | None
    caption: str
    ellipse: tuple | None = None
    logits: np.ndarray | None = None
    region: np.ndarray | None = None
    total: int = 0
    source_to_model: tuple = ()

    comparison: tuple | None = None
    outline: list | None = None
    fit_note: str = ''
    contour: object = None
    source_mask: np.ndarray | None = None
    requested: str = 'ellipse'

    def render(self, threshold, use_legacy=False, show_outside=False, use_contour=True, show_points=False):
        if self.image is None:
            return None, self.caption
        annotated = self.image.copy()
        caption = self.caption
        if self.requested == 'mesh' and self.contour is None:
            return annotated, caption + ' · Mesh unavailable; no ellipse measurement substituted.'
        if not use_contour and not use_legacy and self.region is None:
            return annotated, caption + ' · Landmark ellipse comparison unavailable.'
        if use_legacy and self.comparison is None:
            return annotated, caption + ' · Legacy ellipse comparison unavailable.'
        preview = ('Legacy box ellipse (comparison)' if use_legacy else
                   'Landmark ellipse (comparison)' if not use_contour else 'Measured '+self.requested)
        caption += ' · Preview: '+preview
        if self.logits is not None:
            region, total = self.region, self.total
            if use_contour and self.contour is not None:
                region, total = self.contour.mask, self.contour.total
            if use_legacy and self.comparison:
                region, total, _ = self.comparison
            foreground = foreground_mask(self.logits, region, threshold)
            layers = [(foreground, (0, 240, 120))]
            if show_outside:
                layers.append(((self.logits > threshold) & ~region, (220, 80, 255)))
            for pixels, color in layers:
                with Image.fromarray(pixels.astype('uint8') * 110) as mask:
                    with mask.transform(annotated.size, Image.Transform.AFFINE, self.source_to_model,
                                        resample=Image.Resampling.NEAREST) as alpha:
                        annotated.paste(color, (0, 0, *annotated.size), alpha)
            visible, source = int(foreground.sum()), int(region.sum())
            label = 'contour' if use_contour and not use_legacy and self.contour is not None and self.contour.method == 'mesh' else 'ellipse'
            caption += (f' · Replay {visible / total:.3f} ({visible:,} visible / {total:,} {label} pixels)'
                        f' · Source coverage {source / total:.1%}')
            if self.comparison and self.region is not None:
                alternate, denominator, _ = self.comparison
                base_score = foreground_mask(self.logits,self.region,threshold).sum() / self.total
                alternate_score = foreground_mask(self.logits,alternate,threshold).sum() / denominator
                caption += f' · Face ellipse {base_score:.3f} · Legacy box ellipse {alternate_score:.3f}'
            raw_mask = self.logits > threshold
            if self.source_mask is not None:
                raw_mask &= self.source_mask
            raw = int(raw_mask.sum())
            if raw:
                caption += f' · Predicted mask outside region: {1-visible/raw:.1%}'
            if show_outside:
                caption += ' · Purple: predicted face outside the selected region (not counted)'
        if self.fit_note:
            caption += ' · ' + self.fit_note
        if show_points and self.contour is not None and self.contour.landmarks is not None:
            draw = ImageDraw.Draw(annotated)
            for x,y in self.contour.landmarks:
                draw.ellipse((x-1,y-1,x+1,y+1),fill='cyan')
        if use_contour and not use_legacy and self.contour is not None and self.contour.outline:
            ImageDraw.Draw(annotated).line([tuple(p) for p in self.contour.outline], fill='gold', width=1)
        elif use_legacy and self.comparison:
            ImageDraw.Draw(annotated).line(self.comparison[2], fill='gold', width=1)
        elif self.outline is not None:
            ImageDraw.Draw(annotated).line(self.outline, fill='gold', width=1)
        elif self.ellipse is not None:
            ImageDraw.Draw(annotated).ellipse(self.ellipse, outline='gold', width=1)
        return annotated, caption

    def slider_range(self, configured):
        # Include raw pixels outside the ROI, so its diagnostic overlay can be
        # tuned without rerunning the model or changing the measured region.
        scores = self.logits
        # Include the measured range, the YAML value, and a position below the lowest logit.
        return math.floor(min(float(scores.min()), configured, 0.)) - 1, math.ceil(max(float(scores.max()), configured, 1.))


def replay(previews, address, scope):
    evidence = {}
    image, caption = previews.image('face', address, scope, (None, 'Selected face'), full=True,
                                   face_evidence=evidence)
    if image is None:
        return FaceReplay(None, caption)
    check_cancelled()
    saved = evidence['fraction']
    record = evidence['visibility']
    recorded = f'Recorded {saved:.3f}' if valid_fraction(saved) else 'Recorded fraction unavailable'
    if valid_fraction(saved):
        if cached(dict(visibility=record, visible_face_fraction=saved)):
            recorded += f" at logit {record.get('logit_threshold', DEFAULT_FACEOCC_LOGIT_THRESHOLD):g}"
        else:
            recorded += ' (unknown measurement contract)'
    try:
        face = checked_detection(evidence['geometry'])
    except (TypeError, ValueError, OverflowError):
        return FaceReplay(image, f'{recorded} · Replay unavailable: invalid recorded face geometry.')

    x, y, width, height = face.box
    cx, cy = x + width / 2, y + height / 2
    rx, ry = ellipse_radii(width, height)
    mode = previews.config.data['detection']['visibility_region']
    result = FaceReplay(image, recorded+' · Requested region: '+mode,
                        (cx - rx, cy - ry, cx + rx, cy + ry), requested=mode)
    minimum = previews.config.data['detection']['min_face_source_fraction']
    try:
        if mode == 'ellipse':
            fitted = fit_face_region(face)
            result.outline = face_outline(face,fitted=fitted)
            result.fit_note = f'Landmark fit: {fitted.view}'
            if fitted.ignored:
                result.fit_note += f'; ignored outlier: {fitted.ignored}'
        prepared = prepare_measurement(image, face, minimum, defer_source_check=True, region_mode=mode)
        blob, region, total = prepared
        affine = prepared.affine
    except VisibilityGeometryError as error:
        coverage = error.diagnostics.get('visibility_source_fraction')
        result.caption += f' · Replay unavailable: {error}'
        if coverage is not None:
            result.caption += f' Source coverage {coverage:.1%} (minimum {minimum:.1%}).'
        return result

    check_cancelled()
    try:
        paths = ensure_assets(previews.config, keys=['faceocc'])
        check_cancelled()
        with closing(FaceVisibility(previews.config, paths)) as model:
            from .face_contour import resolve
            result.contour = resolve(model, prepared)
            result.fit_note = (result.fit_note+' · ' if result.fit_note else '')+'Region: '+result.contour.method
            logits = model.inspect_logits(blob)
    except VisibilityGeometryError as error:
        result.caption += f' · Replay unavailable: {error}'
        coverage = error.diagnostics.get('visibility_source_fraction')
        if coverage is not None:
            result.caption += f' Source coverage {coverage:.1%} (minimum {minimum:.1%}).'
        return result
    except FileNotFoundError:
        result.caption += ' · FaceOcc weights missing. Check the model directory and download_missing.'
        return result
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        result.caption += (f' · FaceOcc replay unavailable ({type(error).__name__}). '
                           'Check FaceOcc weights, dependencies and the configured device.')
        return result
    check_cancelled()
    # Pillow maps each displayed source pixel back into the aligned score map.
    result.source_to_model = tuple(np.linalg.inv([affine[:3], affine[3:], (0, 0, 1)])[:2].ravel())
    result.logits, result.region, result.total = logits, region, total
    if mode == 'mesh':
        # Comparison only: an invalid ellipse cannot veto an accepted mesh.
        try:
            fitted = fit_face_region(face)
            comparison = ellipse_roi(fitted.center, fitted.axis, fitted.radii, image.size, affine)
            if comparison is None:
                result.fit_note += '; landmark ellipse comparison extends outside the model crop'
            else:
                result.region, result.total, result.outline = comparison
        except VisibilityGeometryError as error:
            result.fit_note += '; landmark ellipse comparison unavailable: '+str(error)
    from .face_contour import real_pixels
    result.source_mask = real_pixels(affine, image.size)
    result.comparison = legacy_roi(face, image.size, affine)
    if result.comparison is None:
        result.fit_note += '; legacy box extends outside the model crop (comparison unavailable)'
    return result


class FaceVisibilityWindow:
    def __init__(self, host):
        self.host = host
        self.window = tk.Toplevel(host.root)
        self.window.geometry('900x540')
        self.images = InspectorImages(self.window)
        self.images.synchronized = True
        self.images.panels['A'].configure(text='Original cue')
        self.images.panels['B'].configure(text='FaceOcc replay')
        self.images.pack(fill='both', expand=True)
        bar = ttk.Frame(self.window)
        bar.pack(fill='x', padx=6)
        ttk.Button(bar, text='Fit', command=lambda: self.images.fit('A')).pack(side='left')
        ttk.Button(bar, text='100%', command=lambda: self.images.set_zoom('A', 1.)).pack(side='left')
        ttk.Label(bar, text='Gold outline: selected measurement region    Green: pixels counted as visible').pack(side='left')
        region_bar = ttk.Frame(self.window)
        region_bar.pack(fill='x', padx=8)
        ttk.Label(region_bar, text='Region (preview)').pack(side='left', padx=(0,8))
        self.measured_label = 'Measured '+host.previews.config.data['detection']['visibility_region']
        self.region_choice = tk.StringVar(master=self.window, value=self.measured_label)
        self.region_selector = ttk.Combobox(region_bar, textvariable=self.region_choice, state='disabled', width=30)
        self.region_selector.bind('<<ComboboxSelected>>', lambda _: self.set_threshold(self.value.get()))
        self.region_selector.pack(side='left')
        mask_bar = ttk.Frame(self.window)
        mask_bar.pack(fill='x', padx=8)
        self.show_outside = tk.BooleanVar(master=self.window, value=False)
        self.outside_toggle = ttk.Checkbutton(mask_bar, text='Show predicted face outside region (purple)',
                                              variable=self.show_outside,
                                              command=lambda: self.set_threshold(self.value.get()))
        self.outside_toggle.pack(side='left')
        self.show_points = tk.BooleanVar(master=self.window, value=False)
        self.points_toggle = ttk.Checkbutton(region_bar, text='Mesh landmarks', variable=self.show_points,
                        command=lambda: self.set_threshold(self.value.get()))
        self.points_toggle.pack(side='left')
        controls = ttk.Frame(self.window)
        controls.pack(fill='x', padx=8)
        ttk.Label(controls, text='Pixel logit threshold (preview)').pack(side='left')
        self.configured = host.previews.config.data['detection']['faceocc_logit_threshold']
        self.value = tk.DoubleVar(master=self.window, value=self.configured)
        self.slider = ttk.Scale(controls, orient='horizontal', variable=self.value, command=self.slide)
        self.slider.bind('<Left>', lambda _: self.nudge(-0.01))
        self.slider.bind('<Right>', lambda _: self.nudge(0.01))
        self.slider.pack(side='left', fill='x', expand=True, padx=8)
        self.value_label = ttk.Label(controls)
        self.value_label.pack(side='left')
        self.reset = ttk.Button(controls, text='Reset to YAML', command=lambda: self.set_threshold(self.configured))
        self.reset.pack(side='left', padx=8)
        self.result = None
        self.status = ttk.Label(self.window, wraplength=850)
        self.status.pack(anchor='w', padx=8, pady=6)
        self.worker = Worker('inspector-face-visibility', resource=host.previews.fork)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = self.window.after(30, self.poll)

    def show(self, row, scope):
        self.window.title(f"Face visibility · Scene {row['scene']} · cue {row['cue']} · {row['id']}")
        self.window.lift()
        self.result = None
        self.region_selector.configure(state='disabled')
        self.region_choice.set(self.measured_label)
        self.points_toggle.state(['disabled'])
        self.outside_toggle.state(['disabled'])
        self.slider.state(['disabled'])
        self.reset.state(['disabled'])
        self.value.set(self.configured)
        self.value_label.configure(text=str(self.configured))
        self.images.show('A', None, '')
        self.images.show('B', None, '')
        self.status.configure(text='Checking VTT source and replaying FaceOcc…')
        address = row['id']
        self.worker.request(lambda: replay(local.preview, address, scope), self.display, self.failed)

    def display(self, result):
        self.result = result
        self.images.show('A', result.image, '')
        self.images.fit('A')
        if result.logits is not None:
            low, high = result.slider_range(self.configured)
            options = [self.measured_label]
            if result.requested == 'mesh' and result.region is not None:
                options.append('Landmark ellipse (comparison)')
            if result.comparison is not None:
                options.append('Legacy box ellipse (comparison)')
            self.region_selector.configure(values=options, state='readonly')
            if result.contour is not None and result.contour.landmarks is not None:
                self.points_toggle.state(['!disabled'])
            self.outside_toggle.state(['!disabled'])
            self.slider.configure(from_=low, to=high)
            self.slider.state(['!disabled'])
            self.reset.state(['!disabled'])
        self.set_threshold(self.configured)

    def slide(self, value):
        self.set_threshold(round(float(value), 2))

    def nudge(self, step):
        if self.slider.instate(['!disabled']):
            value = round(self.value.get() + step, 2)
            self.set_threshold(max(float(self.slider.cget('from')), min(float(self.slider.cget('to')), value)))
        return 'break'

    def set_threshold(self, threshold):
        self.value.set(threshold)
        self.value_label.configure(text=str(threshold))
        if self.result is not None:
            result = self.result
            selected = self.region_choice.get()
            use_legacy, show_outside = selected == 'Legacy box ellipse (comparison)', self.show_outside.get()
            contour, points = selected == self.measured_label, self.show_points.get()
            self.worker.request(lambda: result.render(threshold, use_legacy, show_outside, contour, points), self.display_overlay, self.failed)

    def display_overlay(self, overlay):
        image, caption = overlay
        self.images.show('B', image, '')
        self.status.configure(text=caption)

    def failed(self, error):
        self.status.configure(text=f'Face visibility preview failed ({type(error).__name__}).')

    def poll(self):
        self.worker.poll()
        self.timer = self.window.after(30, self.poll)

    def close(self):
        self.window.after_cancel(self.timer)
        self.worker.close()
        self.result = None
        self.window.destroy()
        self.host.visual_windows.remove(self)
