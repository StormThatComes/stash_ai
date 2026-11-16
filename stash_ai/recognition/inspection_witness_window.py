"""Navigate pair evidence without replacing the comparison or its selection."""
import tkinter as tk
from tkinter import ttk

from .inspection_context import locate
from .inspection_images import InspectorImages
from .inspection_runtime import Worker
from .inspection_visual import DEFAULT_LAYERS


class WitnessWindow:
    def __init__(self, comparison, evidence):
        self.comparison = comparison
        host = comparison.host
        self.window = tk.Toplevel(comparison.window)
        self.window.title(evidence['label'])
        self.window.geometry('940x520')
        self.images = InspectorImages(self.window)
        self.images.pack(fill='both', expand=True)
        self.images.compare(len(evidence['refs']) > 1)
        ttk.Button(self.window, text='Back to comparison', command=self.close).pack(pady=4)
        self.worker = Worker('pair-evidence', resource=host.previews.fork if host.previews else None)
        scope = comparison.scan['scope']

        def load():
            for kind, address in evidence['refs']:
                row = locate(comparison.reader.config, kind, address, scope)
                preview = host.load_reference(row['reader'], kind, address, scope, full=bool(host.previews),
                                              layers=dict(DEFAULT_LAYERS, masks=True))
                yield row, preview['annotated']

        self.worker.request(lambda: list(load()), self.display, self.failed)
        self.window.protocol('WM_DELETE_WINDOW', self.close)

    def display(self, rows):
        for panel, (row, (image, caption)) in zip(('A', 'B'), rows):
            self.images.panels[panel].configure(text=f"Scene {row['scene']} · cue {row['cue']} · {row['kind']} {row['id']}")
            self.images.show(panel, image, caption)
            self.images.fit(panel)

    def failed(self, error):
        self.images.show('A', None, str(error))

    def close(self):
        self.worker.close()
        self.window.destroy()
        self.comparison.witness_window = None
        self.comparison.window.lift()
