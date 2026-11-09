"""Desktop detection tuner. Tk is imported only for the desktop command."""
from argparse import ArgumentTypeError
from functools import partial
import gc
import os
from threading import Event
import tkinter as tk
from tkinter import ttk

from stash_ai.stash_api.transport import safe_error
from .cli import stash_id
from .config import load
from .detection_session import CueDetector
from .tune_compare import detect_variants
from .tune_images import ImagePair
from .tune_editor import ParameterEditor, Settings, display_value
from .tune_worker import SceneBrowser, Worker


class Tuner:
    def __init__(self, root, config, services, *, detector=None, compare_config=None, device=None):
        self.root = root
        self.browser = SceneBrowser(services, config)
        self.detector = detector or CueDetector(services)
        self.media = Worker('tuner-vtt')
        self.inference = Worker('tuner-detector')
        self.scene_generation = 0
        self.scene_id = None
        self.index = None
        self.frame = None
        self.times = ()
        self.device = device
        self.settings = {'A': Settings(config), 'B': Settings(compare_config or config)}
        self.revisions = {'A': 0, 'B': 0}
        self.cache = {}
        self.stop = Event()
        self.compare = tk.BooleanVar(value=compare_config is not None)
        self.active = tk.StringVar(value='B' if compare_config else 'A')

        root.title('Detection tuner')
        root.geometry('1040x720')
        root.minsize(760, 500)
        self.body = body = ttk.Frame(root, padding=12)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        bar = ttk.Frame(body)
        bar.grid(row=0, column=0, sticky='ew')
        ttk.Label(bar, text='Scene ID').pack(side='left')
        self.scene_entry = ttk.Entry(bar, width=12)
        self.scene_entry.pack(side='left', padx=6)
        self.scene_entry.bind('<Return>', lambda _: self.open_scene())
        ttk.Button(bar, text='Load scene', command=self.open_scene).pack(side='left')
        self.run_button = ttk.Button(bar, text='Run detection', command=self.run_detection, state='disabled')
        self.run_button.pack(side='right')
        ttk.Checkbutton(bar, text='Compare A/B', variable=self.compare,
                        command=self.toggle_compare).pack(side='right', padx=10)

        navigation = ttk.Frame(body)
        navigation.grid(row=1, column=0, sticky='ew', pady=8)
        self.cue_label = tk.StringVar(value='Cue')
        ttk.Label(navigation, textvariable=self.cue_label, width=35).pack(side='left')
        ttk.Button(navigation, text='Previous', command=lambda: self.step(-1)).pack(side='left')
        self.slider = ttk.Scale(navigation, from_=0, to=0, command=self.slide, state='disabled')
        self.slider.pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(navigation, text='Next', command=lambda: self.step(1)).pack(side='left')
        self.slider.bind('<Left>', lambda _: self.step(-1))
        self.slider.bind('<Right>', lambda _: self.step(1))
        self.images = ImagePair(body)
        self.images.grid(row=2, column=0, sticky='nsew')
        tools = ttk.Frame(body)
        tools.grid(row=3, column=0, sticky='ew', pady=8)
        ttk.Label(tools, text='Edit configuration').pack(side='left')
        self.choose = ttk.Combobox(tools, width=4, textvariable=self.active, state='readonly')
        self.choose.pack(side='left', padx=6)
        self.choose.bind('<<ComboboxSelected>>', lambda _: self.choose_editor())
        self.path = ttk.Entry(tools)
        self.path.pack(side='left', fill='x', expand=True)
        self.path.bind('<Return>', lambda _: self.load_yaml())
        ttk.Button(tools, text='Load YAML', command=self.load_yaml).pack(side='left', padx=4)
        self.copy = ttk.Button(tools, text='Copy A to B', command=self.copy_a)
        self.copy.pack(side='left')
        self.diff_button = ttk.Button(tools, text='Differences', command=self.differences)
        self.diff_button.pack(side='left', padx=4)
        ttk.Label(tools, text='Zoom').pack(side='left')
        self.zoom = ttk.Combobox(tools, values=('50%', '100%', '150%', '200%', '400%'),
                                 state='readonly', width=5)
        self.zoom.set('100%')
        self.zoom.pack(side='left', padx=(4, 0))
        self.zoom.bind('<<ComboboxSelected>>', lambda _: self.set_zoom())

        self.editor = ParameterEditor(body, self.settings[self.active.get()], self.edited, self.error)
        self.editor.grid(row=4, column=0, sticky='ew', pady=(0, 8))
        self.status = tk.StringVar(value='Load a scene to browse its VTT cues.')
        ttk.Label(body, textvariable=self.status, wraplength=900).grid(row=5, column=0, sticky='w')
        self.toggle_compare()
        self.scene_entry.focus_set()
        self.poll_id = root.after(35, self.poll)

    def toggle_compare(self):
        enabled = self.compare.get()
        self.images.compare(enabled)
        self.choose.configure(values=self.images.names)
        if not enabled:
            self.active.set('A')
        self.choose_editor()
        for button in (self.copy, self.diff_button):
            button.state(['!disabled' if enabled else 'disabled'])
        if enabled and self.frame is not None and 'B' not in self.cache:
            self.show_raw('B')

    def choose_editor(self):
        settings = self.settings[self.active.get()]
        self.editor.bind_settings(settings)
        self.path.delete(0, 'end')
        if settings.config.source_path:
            self.path.insert(0, str(settings.config.source_path))

    def load_yaml(self):
        try:
            if not self.path.get().strip():
                raise ValueError('Enter the path to a recognition YAML file.')
            config = load(self.path.get().strip(), device=self.device)
            settings = Settings(config)
        except Exception as error:
            self.error(error)
            return
        self.settings[self.active.get()] = settings
        self.choose_editor()
        self.edited()

    def copy_a(self):
        original = self.settings['A']
        self.settings['B'] = Settings(original.config)
        self.settings['B'].edits.update(original.edits)
        self.active.set('B')
        self.choose_editor()
        self.edited()

    def differences(self):
        rows = []
        a, b = self.settings['A'], self.settings['B']
        for path in a.base:
            left, right = a.value(path), b.value(path)
            if left != right:
                rows.append((path, display_value(left), display_value(right)))
        dialog = tk.Toplevel(self.root)
        dialog.title('Detector setting differences')
        if not rows:
            ttk.Label(dialog, text='All detector settings match.', padding=12).pack()
            return
        table = ttk.Treeview(dialog, columns=('Parameter', 'A', 'B'), show='headings')
        for field in table['columns']:
            table.heading(field, text=field)
            table.column(field, width=350 if field == 'Parameter' else 160)
        for row in rows:
            table.insert('', 'end', values=row)
        scroll = ttk.Scrollbar(dialog, orient='vertical', command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        table.pack(fill='both', expand=True)

    def set_zoom(self):
        self.images.zoom = float(self.zoom.get().rstrip('%')) / 100
        self.images.layout()

    def show_raw(self, name, caption='Original VTT cue'):
        self.images.show(name, self.frame.image if self.frame else None, caption if self.frame else '')

    def open_scene(self):
        try:
            scene_id = stash_id(self.scene_entry.get().strip())
        except (ArgumentTypeError, ValueError):
            self.error('Enter a positive integer Scene ID.')
            return
        self.scene_generation += 1
        self.scene_id = scene_id
        self.times = ()
        self.index = None
        self.slider.configure(state='disabled', to=0)
        self.slider.set(0)
        self.select(0)

    def slide(self, value):
        if self.times:
            self.select(round(float(value)))

    def step(self, amount):
        if self.times:
            index = min(max(self.index + amount, 0), len(self.times) - 1)
            self.slider.set(index)
            self.select(index)
        return 'break'

    def select(self, index):
        if self.index == index:
            return
        self.index = index
        self.frame = None
        self.stop.set()
        self.cache.clear()
        self.inference.invalidate()
        self.run_button.state(['disabled'])
        for name in self.settings:
            self.show_raw(name)
        self.cue_label.set(f'Cue {index}')
        self.status.set('Loading VTT preview...')
        self.media.request(partial(self.browser.preview, self.scene_generation, self.scene_id, index),
                           self.preview_ready, self.error)

    def preview_ready(self, preview):
        self.times, self.frame = preview
        self.slider.configure(to=len(self.times)-1, state='normal' if len(self.times) > 1 else 'disabled')
        cue = self.frame.cue
        self.cue_label.set(f'Cue {self.index} / {len(self.times)-1}   {cue.start:.2f}-{cue.end:.2f} s')
        for name in self.images.names:
            self.show_raw(name)
        self.run_button.state(['!disabled'])
        self.status.set('Ready')

    def edited(self):
        name = self.active.get()
        self.revisions[name] += 1
        self.cache.pop(name, None)
        self.show_raw(name)
        if self.frame is not None:
            self.run_button.state(['!disabled'])
        self.status.set('Settings changed. Run detection to see the result.')

    def run_detection(self):
        if self.frame is None:
            return
        plans = {}
        for name in self.images.names:
            if name not in self.cache:
                try:
                    plans[name] = self.settings[name].effective()
                except (ValueError, TypeError) as error:
                    self.show_raw(name, 'Settings error: ' + str(error))
        if not plans:
            self.status.set('Results are current.' if all(n in self.cache for n in self.images.names)
                            else 'Check the settings in the indicated panel.')
            return
        self.stop.set()
        self.stop = Event()
        self.run_button.state(['disabled'])
        self.status.set('Running detection for ' + ', '.join(plans) + '...')
        operation = partial(detect_variants, self.detector, self.frame, plans,
                            list(self.cache.values()), self.stop)
        completed = partial(self.detected, plans, self.revisions.copy())
        self.inference.request(operation, completed, self.error)

    def detected(self, plans, revisions, outputs):
        for name, result in outputs.items():
            if revisions[name] != self.revisions[name]:
                continue
            if isinstance(result, str):
                self.show_raw(name, 'Detection failed: ' + result)
                continue
            self.cache[name] = (plans[name], result)
            image, record = result
            matches = sum(face['person_index'] is not None for face in record['faces'])
            self.images.show(name, image, f"Faces: {len(record['faces'])}    People: {len(record['people'])}"
                                          f"    Matched faces: {matches}")
        self.run_button.state(['!disabled'])
        self.status.set('Detection complete' if all(n in self.cache for n in self.images.names)
                        else 'One or more panels have no current result.')

    def error(self, error):
        self.status.set(safe_error(error, os.getenv('STASH_API_KEY', '')))
        if self.frame is not None:
            self.run_button.state(['!disabled'])

    def poll(self):
        self.media.poll()
        self.inference.poll()
        self.poll_id = self.root.after(35, self.poll)

    def close(self):
        self.root.after_cancel(self.poll_id)
        self.stop.set()
        try:
            try:
                self.media.close(self.browser.close)
            finally:
                self.inference.close(self.detector.close)
        finally:
            self.body.destroy()


def execute(args, config, *, services=None):
    from .run import Services
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError('The detection tuner requires a graphical desktop and working Tcl/Tk. '
                           'Try python -m tkinter in this environment.') from error
    try:
        compare = load(args.compare_config, device=args.device) if args.compare_config else None
        window = Tuner(root, config, services or Services(), compare_config=compare, device=args.device)
        root.protocol('WM_DELETE_WINDOW', root.quit)
        try:
            if args.scene_id:
                window.scene_entry.insert(0, args.scene_id)
                window.open_scene()
            root.mainloop()
        finally:
            root.withdraw()
            window.close()
            del window
    finally:
        root.destroy()
        # Finalize cycles containing Tk variables/images on the thread owning Tcl.
        gc.collect()
    return 0
