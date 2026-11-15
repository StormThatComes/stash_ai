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
from .visibility import (FaceVisibility, VisibilityGeometryError, cached, ellipse_radii, foreground_mask,
                         prepare_aligned, valid_fraction)


@dataclass
class FaceReplay:
    image: Image.Image | None
    caption: str
    ellipse: tuple | None = None
    logits: np.ndarray | None = None
    region: np.ndarray | None = None
    total: int = 0
    source_to_model: tuple = ()

    def render(self, threshold):
        if self.ellipse is None:
            return None, self.caption
        annotated = self.image.copy()
        caption = self.caption
        if self.logits is not None:
            foreground = foreground_mask(self.logits, self.region, threshold)
            with Image.fromarray(foreground.astype('uint8') * 110) as mask:
                with mask.transform(annotated.size, Image.Transform.AFFINE, self.source_to_model,
                                    resample=Image.Resampling.NEAREST) as alpha:
                    annotated.paste((0, 240, 120), (0, 0, *annotated.size), alpha)
            visible, source = int(foreground.sum()), int(self.region.sum())
            caption += (f' · Replay {visible / self.total:.3f} ({visible:,} visible / {self.total:,} ellipse pixels)'
                        f' · Source coverage {source / self.total:.1%}')
        ImageDraw.Draw(annotated).ellipse(self.ellipse, outline='gold', width=1)
        return annotated, caption

    def slider_range(self, configured):
        scores = self.logits[self.region]
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
    result = FaceReplay(image, recorded, (cx - rx, cy - ry, cx + rx, cy + ry))
    minimum = previews.config.data['detection']['min_face_source_fraction']
    try:
        blob, region, total, affine = prepare_aligned(image, face, minimum)
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
            logits = model.inspect_logits(blob)
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
        ttk.Label(bar, text='Gold outline: measurement ellipse    Green: pixels counted as visible').pack(side='left')
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
            self.worker.request(lambda: result.render(threshold), self.display_overlay, self.failed)

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
