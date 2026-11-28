"""Candidate comparison on the active cue. Only Tk's owner thread publishes results."""
from functools import partial
import tkinter as tk
from tkinter import ttk

from .tune_images import ImagePair
from .tune_mask_candidate import LOCAL_KEYS, inspect, preview
from .tune_worker import Worker


class LiveMasks(ttk.Frame):
    def __init__(self, parent, verify):
        super().__init__(parent)
        self.active = 'A'
        self.visible = False
        self.samples = {}
        self.settings = {}
        self.indices = {}
        self.selected_attempt = {}
        self.details = {}
        self.omitted = {}
        self.tokens = {}
        self.source = None
        self.workers = {name: Worker('cleanup-' + name) for name in ('A', 'B')}
        bar = ttk.Frame(self)
        bar.pack(fill='x', pady=4)
        self.choice = ttk.Combobox(bar, state='readonly', width=32)
        self.choice.pack(side='left', fill='x', expand=True)
        self.choice.bind('<<ComboboxSelected>>', self.select)
        self.view = ttk.Combobox(bar, state='readonly', width=15, values=('Raw', 'Cleaned', 'Removed pixels'))
        self.view.set('Removed pixels')
        self.view.pack(side='left')
        self.view.bind('<<ComboboxSelected>>', lambda _: self.refresh())
        ttk.Button(bar, text='Check full detection', command=verify).pack(side='left')
        self.images = ImagePair(self)
        self.images.pack(fill='both', expand=True)
        self.info = tk.StringVar()
        ttk.Label(self, textvariable=self.info, wraplength=700).pack(anchor='w')
        for name, panel in self.images.panels.items():
            panel.canvas.bind('<Button-1>', partial(self.clicked, name))

    def clicked(self, name, event):
        data = self.details.get(name)
        if data is not None:
            panel = self.images.panels[name]
            x, y = panel.canvas.canvasx(event.x), panel.canvas.canvasy(event.y)
            self.info.set(inspect(*data, int(x/panel.zoom), int(y/panel.zoom)))

    def select(self, _):
        self.indices[self.active] = self.choice.current()
        self.selected_attempt[self.active] = self.samples[self.active][self.choice.current()].attempt_key
        self.request(self.active)

    def configure_panel(self, name, compare):
        self.active = name
        self.images.compare(compare)
        self.choices()
        if self.visible:
            self.refresh()

    def choices(self):
        samples = self.samples.get(self.active, ())
        self.choice.configure(values=[sample.title for sample in samples])
        if samples:
            self.choice.current(self.indices[self.active])
        else:
            self.choice.set('')

    def clear(self, name, *, keep_token=False):
        if not keep_token:
            self.tokens.pop(name, None)
            self.selected_attempt.pop(name, None)
        self.workers[name].invalidate()
        self.samples.pop(name, None)
        self.settings.pop(name, None)
        self.details.pop(name, None)
        self.indices.pop(name, None)
        self.images.show(name, None, 'Run detection to collect mask candidates.')
        self.info.set('')
        self.choices()

    def set_samples(self, name, history, settings, source):
        token = getattr(history, 'replay_token', None)
        self.clear(name, keep_token=token is not None and token is self.tokens.get(name))
        self.tokens[name] = token
        self.source = source
        self.samples[name] = getattr(history, 'samples', ())
        self.omitted[name] = getattr(history, 'samples_omitted', 0)
        key = self.selected_attempt.get(name)
        self.indices[name] = next((i for i, sample in enumerate(self.samples[name]) if sample.attempt_key == key), 0)
        if self.samples[name]:
            self.selected_attempt[name] = self.samples[name][self.indices[name]].attempt_key
        self.settings[name] = settings
        self.choices()
        self.request(name)

    def activate(self, visible):
        self.visible = visible
        if visible:
            self.refresh()
        else:
            for worker in self.workers.values():
                worker.invalidate()

    def refresh(self):
        for name in self.images.names:
            self.request(name)

    def request(self, name):
        if not self.visible or name not in self.images.names:
            return
        self.info.set('')
        self.details.pop(name, None)
        if not self.samples.get(name):
            self.images.show(name, None, 'No recorded candidate. Check mask eligibility and history limits.')
            return
        config = self.settings[name]
        settings = {k: config.value('pose.mask.' + k) for k in LOCAL_KEYS}
        settings['expand_boxes'] = config.value('pose.mask.apply_to.box_expansion')
        sample = self.samples[name][self.indices[name]]
        self.workers[name].request(partial(preview, self.source, sample, settings, self.view.get()),
                                   partial(self.complete, name), partial(self.failed, name))

    def complete(self, name, output):
        image, caption, details, cleaned = output
        self.details[name] = details, cleaned
        if self.omitted[name]:
            caption += f' | {self.omitted[name]} candidates omitted'
        self.images.show(name, image, caption)

    def failed(self, name, error):
        self.images.show(name, None, str(error))

    def poll(self):
        for worker in self.workers.values():
            worker.poll()

    def close(self):
        for worker in self.workers.values():
            worker.close(lambda: None)
