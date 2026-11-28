"""English detector parameter editor backed by the public recognition YAML layout."""
import math
import re
import tkinter as tk

import yaml
from tkinter import ttk

from .config import defaults
from .config_help import HELP
from .config_layout import POINT_FIELDS, canonical_path, get, put, leaf_items, v2_document
from .detection_parameters import apply_edits
from .tune_sharpness import EDITABLE


PMPOSE_SWITCHES = ('pose.deduplication.enabled',)

CHOICES = {
    'face.quality.sharpness_mode': ('legacy', 'split'),
    'body.collection.sharpness_mode': ('legacy', 'split'),
    'face.detector.backend': ('scrfd_10g', 'scrfd_2_5g', 'yunet', 'retinaface_r50'),
    'pose.detector.backend': ('rfdetr_xl', 'dfine_x', 'dfine_l', 'yolo', 'rtmo'),
    'pose.estimator.backend': ('vitpose_plus_large', 'pmpose', 'sapiens', 'yolo', 'rtmo'),
    'pose.estimator.sapiens.variant': ('0.3b', '0.6b', '1b'),
    'pose.mask.backend': ('sam_pose2seg', 'sam3', 'hq_sam2_large', 'rfdetr_seg_large'),
    'pose.mask.mode': ('all', 'overlap'),
}


def display_value(value):
    if value is None:
        return 'null'
    return str(value).lower() if isinstance(value, bool) else str(value)


class Settings:
    def __init__(self, config):
        self.config = config
        document = v2_document(config.data, source=config.source)
        self.base = {path: value for path, value in leaf_items(document)
                     if path.startswith(('face.detector.', 'pose.')) or path in EDITABLE}
        # An operator may write 1 for a real-valued threshold. Parse according to
        # the schema so that threshold can still be changed to (for example) 0.8.
        self.schema = v2_document(defaults())
        self.edits = {}
        self.undo_stack = []

    def value(self, path):
        path = canonical_path(path)
        if path in PMPOSE_SWITCHES and self.value('pose.estimator.backend') not in ('pmpose','vitpose_plus_large','sapiens'):
            return False
        return self.edits.get(path, self.base[path])

    def edit(self, path, text):
        path = canonical_path(path)
        original = self.base[path]
        default = get(self.schema, path)
        if isinstance(default, str):
            value = text
        elif isinstance(default, bool):
            if text not in ('true', 'false'):
                raise ValueError(f'{path} requires true or false.')
            value = text == 'true'
        elif text.lower() in ('null', 'auto') and nullable(path):
            value = None
        else:
            try:
                value = int(text) if type(default) is int else float(text)
            except ValueError as error:
                raise ValueError(f'Enter a number for {path}.') from error
            if not math.isfinite(value):
                raise ValueError(f'Enter a finite number for {path}.')
            limits = bounds(path)
            if limits and not limits[0] <= value <= limits[1]:
                raise ValueError(f'{path} must be in {limits[0]:g}..{limits[1]:g}.')
        if path in ('pose.detector.backend','pose.estimator.backend'):
            peer='pose.estimator.backend' if path=='pose.detector.backend' else 'pose.detector.backend'
            if value=='rtmo':self.edits[peer]='rtmo'
            elif self.value(peer)=='rtmo':
                self.edits[peer]='yolo' if value=='yolo' else ('vitpose_plus_large' if peer=='pose.estimator.backend' else 'rfdetr_xl')
        if original == value:
            self.edits.pop(path, None)
        else:
            self.edits[path] = value

    def differences(self):
        return {path: value for path, baseline in self.base.items()
                 if (value := self.value(path)) != baseline}

    def effective(self):
        return apply_edits(self.config, self.differences(), detection_preview=True)[0]

    def groups(self):
        box = self.value('pose.detector.backend')
        estimator = self.value('pose.estimator.backend')
        allowed = ['face.detector', 'face.quality', 'body.collection', 'pose.detector',
                   'pose.detector.' + ('dfine' if box.startswith('dfine') else box),
                   'pose.estimator', 'pose.estimator.' + estimator]
        if estimator=='rtmo':allowed.extend(('pose.point_quality','pose.mask'))
        if estimator in ('sapiens','vitpose_plus_large'):
            allowed.extend(('pose.point_quality','pose.mask','pose.deduplication'))
        if estimator == 'pmpose':
            allowed.extend(('pose.point_quality', 'pose.mask', 'pose.deduplication'))
        if 'pose.mask' in allowed:
            allowed.append('pose.mask.apply_to')
        available = {key.rpartition('.')[0] for key in self.base}
        return [group for group in allowed if group in available]


def nullable(path):
    return (path in ('pose.mask.min_recovery_mask_score', 'face.detector.candidate_score_threshold') or
            path.startswith(('pose.mask.', 'pose.deduplication.')) and path.rpartition('.')[2] in POINT_FIELDS)


def bounds(path):
    match = re.search(r'(?:Range: |\()(\d+(?:\.\d+)?)(?:\.\.|-)(\d+(?:\.\d+)?)', HELP.get(path, ''))
    return tuple(map(float, match.groups())) if match else None


class ParameterEditor(ttk.Frame):
    def __init__(self, parent, settings, changed, failed):
        super().__init__(parent, padding=6)
        self.changed, self.failed = changed, failed
        self.settings = settings
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.query = tk.StringVar()
        self.only_changed = tk.BooleanVar()
        ttk.Label(self, text='Find parameter').grid(row=0, column=0, sticky='w')
        ttk.Entry(self, textvariable=self.query).grid(row=1, column=0, sticky='ew')
        table = ttk.Frame(self)
        table.grid(row=2, column=0, sticky='nsew', pady=4)
        self.tree = ttk.Treeview(table, columns=('value',), show='tree headings', selectmode='browse', height=10)
        self.tree.heading('#0', text='Parameter')
        self.tree.heading('value', text='Value')
        self.tree.column('#0', width=188, minwidth=85)
        self.tree.column('value', width=60, minwidth=35, stretch=False)
        bar = ttk.Scrollbar(table, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True)
        self.tree.tag_configure('changed', foreground='#235fa3')
        self.tree.bind('<<TreeviewSelect>>', lambda _: self.show_value())
        ttk.Checkbutton(self, text='Changed only', variable=self.only_changed, command=self.refresh).grid(
            row=3, column=0, sticky='w')
        self.manual = tk.StringVar()
        self.value_input = ttk.Combobox(self, textvariable=self.manual)
        self.value_input.grid(row=4, column=0, sticky='ew')
        self.value_input.bind('<Return>', lambda _: self.apply())
        self.value_input.bind('<FocusOut>', lambda _: self.apply() if self.path == self.selected() else None)
        self.value_input.bind('<<ComboboxSelected>>', lambda _: self.apply())
        self.flag = tk.BooleanVar()
        self.toggle = ttk.Checkbutton(self, variable=self.flag, text='Enabled',
                                     command=lambda: self.commit(display_value(self.flag.get())))
        self.toggle.grid(row=4, column=0, sticky='w')
        self.position = tk.DoubleVar()
        self.scale = ttk.Scale(self, from_=0, to=1, variable=self.position, command=self.slide)
        self.scale.grid(row=5, column=0, sticky='ew', pady=4)
        self.scale.bind('<ButtonPress-1>', lambda _: setattr(self, 'drag_start', self.settings.edits.copy()))
        self.scale.bind('<ButtonRelease-1>', self.end_drag)
        self.drag_start = None
        actions = ttk.Frame(self)
        actions.grid(row=6, column=0, sticky='ew')
        self.auto = ttk.Button(actions, text='Auto', width=5, command=lambda: self.commit('null'))
        self.auto.pack(side='left')
        ttk.Button(actions, text='Reset', width=6, command=self.reset).pack(side='left')
        ttk.Button(actions, text='Undo', width=6, command=self.undo).pack(side='left')
        ttk.Button(actions, text='Copy YAML diff', command=self.copy_diff).pack(side='left')
        self.help = tk.StringVar()
        ttk.Label(self, textvariable=self.help, wraplength=255).grid(row=7, column=0, sticky='ew', pady=4)
        self.path = None
        self.query.trace_add('write', lambda *_: self.refresh())
        self.bind_settings(settings)

    def bind_settings(self, settings):
        self.settings = settings
        self.drag_start = None
        self.refresh()

    def refresh(self):
        selected = self.path
        opened = {i for i in self.tree.get_children() if self.tree.item(i, 'open')}
        self.tree.delete(*self.tree.get_children())
        query = self.query.get().casefold().strip()
        groups = self.settings.groups()
        first = None
        for path, base in self.settings.base.items():
            group, _, name = path.rpartition('.')
            value = self.settings.value(path)
            if group not in groups or query not in path.casefold() or self.only_changed.get() and value == base:
                continue
            if not self.tree.exists(group):
                self.tree.insert('', 'end', iid=group, text=group, open=bool(query or group in opened))
            self.tree.insert(group, 'end', iid=path, text=name, values=(display_value(value),),
                             tags=('changed',) if value != base else ())
            first = first or path
        choice = selected if selected and self.tree.exists(selected) else first
        if choice:
            self.tree.selection_set(choice)
            self.tree.see(choice)
        self.show_value()

    def selected(self):
        rows = self.tree.selection()
        return rows[0] if rows and rows[0] in self.settings.base else None

    def show_value(self):
        self.path = path = self.selected()
        self.toggle.grid_remove()
        self.scale.grid_remove()
        self.value_input.grid_remove()
        self.auto.state(['!disabled' if path and nullable(path) else 'disabled'])
        self.help.set(HELP.get(path, ''))
        if not path:
            return
        value = self.settings.value(path)
        if isinstance(value, bool):
            self.flag.set(value)
            self.toggle.grid()
        else:
            choices = CHOICES.get(path, ())
            self.value_input.configure(values=choices, state='readonly' if choices else 'normal')
            self.manual.set(display_value(value))
            self.value_input.grid()
            limits = bounds(path)
            if limits and not choices:
                lo, hi = limits
                self.power = 3 if hi-lo > 100 or path.endswith('island_fraction') else 1
                fraction = 0 if value is None else min(1, max(0, (value-lo)/(hi-lo)))
                self.position.set(fraction ** (1/self.power))
                self.scale.grid()

    def slide(self, position):
        if self.path:
            lo, hi = bounds(self.path)
            value = lo + (hi-lo) * float(position)**self.power
            integer = type(get(self.settings.schema, self.path)) is int
            self.commit(str(round(value) if integer else round(value, 6)))

    def end_drag(self, _):
        if self.drag_start is not None and self.drag_start != self.settings.edits:
            self.remember(self.drag_start)
        self.drag_start = None

    def remember(self, edits):
        self.settings.undo_stack.append(edits)
        del self.settings.undo_stack[:-100]

    def commit(self, text):
        if not self.path:
            return
        previous = self.settings.edits.copy()
        try:
            self.settings.edit(self.path, text)
        except ValueError as error:
            self.failed(error)
            return
        if previous == self.settings.edits:
            return
        if self.drag_start is None:
            self.remember(previous)
        path = self.path
        if path.endswith('.backend') or self.only_changed.get():
            self.refresh()
        else:
            value = self.settings.value(path)
            self.tree.item(path, values=(display_value(value),),
                           tags=('changed',) if value != self.settings.base[path] else ())
            self.show_value()
        self.changed(path)

    def apply(self):
        if self.path and not isinstance(self.settings.value(self.path), bool):
            self.commit(self.manual.get().strip())

    def reset(self):
        if self.path:
            self.commit(display_value(self.settings.base[self.path]))

    def undo(self):
        if self.settings.undo_stack:
            before = self.settings.edits
            self.settings.edits = self.settings.undo_stack.pop()
            paths = [p for p in before.keys() | self.settings.edits.keys()
                     if before.get(p, self.settings.base[p]) != self.settings.value(p)]
            self.refresh()
            self.changed(paths[0] if len(paths) == 1 else None)

    def copy_diff(self):
        document = {}
        for path, value in self.settings.differences().items():
            put(document, path, value)
        self.clipboard_clear()
        self.clipboard_append(yaml.safe_dump(document, sort_keys=False))
