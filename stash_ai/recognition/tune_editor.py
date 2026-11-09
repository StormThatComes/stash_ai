"""English detector parameter editor backed by the public recognition YAML layout."""
import tkinter as tk
from tkinter import ttk

from .config import defaults
from .config_help import HELP
from .config_layout import POINT_FIELDS, get, leaf_items, v2_document
from .detection_parameters import apply_edits


CHOICES = {
    'face.detector.backend': ('scrfd_10g', 'scrfd_2_5g', 'yunet'),
    'pose.detector.backend': ('rfdetr_xl', 'dfine_x', 'dfine_l', 'yolo'),
    'pose.estimator.backend': ('vitpose_plus_large', 'pmpose', 'sapiens', 'yolo'),
    'pose.estimator.sapiens.variant': ('0.3b', '0.6b', '1b'),
    'pose.refinement.mode': ('all', 'overlap'),
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
                     if path.startswith(('face.detector.', 'pose.'))}
        # An operator may write 1 for a real-valued threshold. Parse according to
        # the schema so that threshold can still be changed to (for example) 0.8.
        self.schema = v2_document(defaults())
        self.edits = {}

    def value(self, path):
        return self.edits.get(path, self.base[path])

    def edit(self, path, text):
        original = self.base[path]
        default = get(self.schema, path)
        if isinstance(default, str):
            value = text
        elif isinstance(default, bool):
            if text not in ('true', 'false'):
                raise ValueError(f'{path} requires true or false.')
            value = text == 'true'
        elif (text == 'null' and path.startswith(('pose.refinement.', 'pose.deduplication.'))
              and path.rpartition('.')[2] in POINT_FIELDS):
            value = None
        else:
            try:
                value = int(text) if type(default) is int else float(text)
            except ValueError as error:
                raise ValueError(f'Enter a number for {path}.') from error
        if original == value:
            self.edits.pop(path, None)
        else:
            self.edits[path] = value

    def effective(self):
        return apply_edits(self.config, self.edits)[0]

    def groups(self):
        box = self.value('pose.detector.backend')
        estimator = self.value('pose.estimator.backend')
        allowed = ['face.detector', 'pose.detector',
                   'pose.detector.' + ('dfine' if box.startswith('dfine') else box),
                   'pose.estimator', 'pose.estimator.' + estimator]
        if estimator == 'pmpose':
            allowed.extend(('pose.point_quality', 'pose.refinement', 'pose.deduplication'))
        # Keep the enable switches reachable when changing away from PMPose.
        for stage in ('pose.refinement', 'pose.deduplication'):
            if self.value(stage + '.enabled') and stage not in allowed:
                allowed.append(stage)
        available = {key.rpartition('.')[0] for key in self.base}
        return [group for group in allowed if group in available]


class ParameterEditor(ttk.Frame):
    def __init__(self, parent, settings, changed, failed):
        super().__init__(parent)
        self.changed, self.failed = changed, failed
        self.group = ttk.Combobox(self, state='readonly', width=28)
        self.parameter = ttk.Combobox(self, state='readonly', width=26)
        self.value_input = ttk.Combobox(self, width=20)
        for column, (name, widget) in enumerate((('Group', self.group), ('Parameter', self.parameter),
                                                  ('Value', self.value_input))):
            ttk.Label(self, text=name).grid(row=0, column=column, sticky='w')
            widget.grid(row=1, column=column, sticky='ew', padx=(0, 6))
            self.columnconfigure(column, weight=1)
        ttk.Button(self, text='Set value', command=self.apply).grid(row=1, column=3)
        self.help = tk.StringVar()
        ttk.Label(self, textvariable=self.help, wraplength=800).grid(
            row=2, column=0, columnspan=4, sticky='w', pady=(4, 0))
        self.group.bind('<<ComboboxSelected>>', lambda _: self.show_group())
        self.parameter.bind('<<ComboboxSelected>>', lambda _: self.show_value())
        self.value_input.bind('<Return>', lambda _: self.apply())
        self.bind_settings(settings)

    def bind_settings(self, settings):
        self.settings = settings
        groups = settings.groups()
        self.group.configure(values=groups)
        if self.group.get() not in groups:
            self.group.set(groups[0])
        self.show_group()

    def show_group(self):
        paths = [path.rpartition('.')[2] for path in self.settings.base
                 if path.rpartition('.')[0] == self.group.get()]
        self.parameter.configure(values=paths)
        if self.parameter.get() not in paths:
            self.parameter.set(paths[0])
        self.show_value()

    def show_value(self):
        path = self.group.get() + '.' + self.parameter.get()
        value = self.settings.value(path)
        choices = CHOICES.get(path) or (('true', 'false') if isinstance(value, bool) else ())
        self.value_input.configure(values=choices, state='readonly' if choices else 'normal')
        self.value_input.set(display_value(value))
        self.help.set(HELP.get(path, ''))

    def apply(self):
        path = self.group.get() + '.' + self.parameter.get()
        try:
            self.settings.edit(path, self.value_input.get().strip())
        except ValueError as error:
            self.failed(error)
            return
        self.bind_settings(self.settings)
        self.changed()
