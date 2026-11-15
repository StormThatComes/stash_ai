"""On-demand comparison of a retained face score with FaceOcc on its verified cue."""
from contextlib import closing
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw

from .inspection_images import InspectorImages
from .inspection_runtime import Worker, check_cancelled, local
from .models import checked_detection, ensure_assets
from .visibility import (CONTRACT_ID, FaceVisibility, VisibilityGeometryError, ellipse_radii,
                         prepare_aligned, valid_fraction)


def replay(previews, address, scope):
    evidence = {}
    image, caption = previews.image('face', address, scope, (None, 'Selected face'), full=True,
                                   face_evidence=evidence)
    if image is None:
        return None, None, caption
    check_cancelled()
    saved = evidence['fraction']
    recorded = f'Recorded {saved:.3f}' if valid_fraction(saved) else 'Recorded fraction unavailable'
    if valid_fraction(saved) and evidence['contract'] != CONTRACT_ID:
        recorded += ' (different or unknown measurement contract)'
    try:
        face = checked_detection(evidence['geometry'])
    except (TypeError, ValueError, OverflowError):
        return image, None, f'{recorded} · Replay unavailable: invalid recorded face geometry.'

    annotated = image.copy()
    result = _overlay(annotated, image, face, previews.config)
    x, y, width, height = face.box
    cx, cy = x + width / 2, y + height / 2
    rx, ry = ellipse_radii(width, height)
    ImageDraw.Draw(annotated).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline='gold', width=1)
    return image, annotated, f'{recorded} · {result}'


def _overlay(annotated, image, face, config):
    minimum = config.data['detection']['min_face_source_fraction']
    try:
        blob, region, total, affine = prepare_aligned(image, face, minimum)
    except VisibilityGeometryError as error:
        coverage = error.diagnostics.get('visibility_source_fraction')
        detail = f' Source coverage {coverage:.1%} (minimum {minimum:.1%}).' if coverage is not None else ''
        return f'Replay unavailable: {error}{detail}'

    check_cancelled()
    try:
        paths = ensure_assets(config, keys=['faceocc'])
        check_cancelled()
        with closing(FaceVisibility(config, paths)) as model:
            foreground = model.inspect_foreground(blob, region)
    except FileNotFoundError:
        return 'FaceOcc weights missing. Check the model directory and download_missing.'
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        return (f'FaceOcc replay unavailable ({type(error).__name__}). '
                'Check FaceOcc weights, dependencies and the configured device.')
    check_cancelled()
    # Pillow needs the inverse mapping: each source pixel samples the aligned mask.
    source_to_model = np.linalg.inv([affine[:3], affine[3:], (0, 0, 1)])[:2].ravel()
    with Image.fromarray(foreground.astype('uint8') * 110) as mask:
        with mask.transform(image.size, Image.Transform.AFFINE, source_to_model,
                            resample=Image.Resampling.NEAREST) as alpha:
            annotated.paste((0, 240, 120), (0, 0, *image.size), alpha)
    visible, source = int(foreground.sum()), int(region.sum())
    return (f'Replay {visible / total:.3f} ({visible:,} visible / {total:,} ellipse pixels) · '
            f'Source coverage {source / total:.1%}')


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
        self.status = ttk.Label(self.window, wraplength=850)
        self.status.pack(anchor='w', padx=8, pady=6)
        self.worker = Worker('inspector-face-visibility', resource=host.previews.fork)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = self.window.after(30, self.poll)

    def show(self, row, scope):
        self.window.title(f"Face visibility · Scene {row['scene']} · cue {row['cue']} · {row['id']}")
        self.window.lift()
        self.images.show('A', None, '')
        self.images.show('B', None, '')
        self.status.configure(text='Checking VTT source and replaying FaceOcc…')
        address = row['id']
        self.worker.request(lambda: replay(local.preview, address, scope), self.display, self.failed)

    def display(self, result):
        original, overlay, status = result
        self.images.show('A', original, '')
        self.images.show('B', overlay, '')
        self.images.fit('A')
        self.status.configure(text=status)

    def failed(self, error):
        self.status.configure(text=f'Face visibility preview failed ({type(error).__name__}).')

    def poll(self):
        self.worker.poll()
        self.timer = self.window.after(30, self.poll)

    def close(self):
        self.window.after_cancel(self.timer)
        self.worker.close()
        self.window.destroy()
        self.host.visual_windows.remove(self)
