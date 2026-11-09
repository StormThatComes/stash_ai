"""Desktop detection tuner. Tk is imported only for the desktop command."""
from functools import partial
import os
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from stash_ai.stash_api.transport import safe_error
from .cli import stash_id
from .detection_session import CueDetector
from .tune_editor import ParameterEditor, Settings
from .tune_worker import SceneBrowser, Worker


class ImagePanel(ttk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, padding=5)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, background='#20242a', highlightthickness=0, width=320, height=220)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.item = self.canvas.create_image(0, 0, anchor='nw')
        x = ttk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        y = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        x.grid(row=1, column=0, sticky='ew')
        y.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(xscrollcommand=x.set, yscrollcommand=y.set)
        self.caption = tk.StringVar()
        ttk.Label(self, textvariable=self.caption).grid(row=2, column=0, sticky='w')
        self.photo = None

    def show(self, image, caption):
        self.photo = ImageTk.PhotoImage(image, master=self.canvas) if image is not None else None
        self.canvas.itemconfigure(self.item, image=self.photo or '')
        self.canvas.configure(scrollregion=(0, 0, *(image.size if image is not None else (0, 0))))
        self.caption.set(caption)


class Tuner:
    def __init__(self, root, config, services, *, detector=None):
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
        self.settings = Settings(config)

        root.title('Detection tuner')
        root.geometry('1040x720')
        root.minsize(760, 500)
        body = ttk.Frame(root, padding=12)
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

        navigation = ttk.Frame(body)
        navigation.grid(row=1, column=0, sticky='ew', pady=8)
        self.cue_label = tk.StringVar(value='Cue')
        ttk.Label(navigation, textvariable=self.cue_label, width=30).pack(side='left')
        ttk.Button(navigation, text='Previous', command=lambda: self.step(-1)).pack(side='left')
        self.slider = ttk.Scale(navigation, from_=0, to=0, command=self.slide, state='disabled')
        self.slider.pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(navigation, text='Next', command=lambda: self.step(1)).pack(side='left')
        self.slider.bind('<Left>', lambda _: self.step(-1))
        self.slider.bind('<Right>', lambda _: self.step(1))
        self.panel = ImagePanel(body, 'Detection preview')
        self.panel.grid(row=2, column=0, sticky='nsew')
        self.editor = ParameterEditor(body, self.settings, self.edited, self.error)
        self.editor.grid(row=3, column=0, sticky='ew', pady=8)
        self.status = tk.StringVar(value='Load a scene to browse its VTT cues.')
        ttk.Label(body, textvariable=self.status, wraplength=900).grid(row=4, column=0, sticky='w')
        self.scene_entry.focus_set()
        self.poll_id = root.after(35, self.poll)

    def open_scene(self):
        try:
            scene_id = stash_id(self.scene_entry.get().strip())
        except ValueError:
            self.error('Enter a positive integer Scene ID.')
            return
        except Exception as error:
            self.error(error)
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
        self.inference.invalidate()
        self.run_button.state(['disabled'])
        self.panel.show(None, '')
        self.cue_label.set(f'Cue {index}')
        self.status.set('Loading VTT preview…')
        self.media.request(partial(self.browser.preview, self.scene_generation, self.scene_id, index),
                           self.preview_ready, self.error)

    def preview_ready(self, preview):
        self.times, self.frame = preview
        self.slider.configure(to=len(self.times)-1, state='normal' if len(self.times) > 1 else 'disabled')
        cue = self.frame.cue
        self.cue_label.set(f'Cue {self.index} / {len(self.times)-1}   {cue.start:.2f}–{cue.end:.2f} s')
        self.panel.show(self.frame.image, 'Original VTT cue')
        self.run_button.state(['!disabled'])
        self.status.set('Ready')

    def edited(self):
        self.inference.invalidate()
        if self.frame is not None:
            self.panel.show(self.frame.image, 'Original VTT cue')
            self.run_button.state(['!disabled'])
        self.status.set('Settings changed. Run detection to see the result.')

    def run_detection(self):
        if self.frame is None:
            return
        try:
            config = self.settings.effective()
        except ValueError as error:
            self.error(error)
            return
        self.run_button.state(['disabled'])
        self.status.set('Running detection…')
        self.inference.request(partial(self.detector.render, self.frame, config), self.detected, self.error)

    def detected(self, output):
        image, record = output
        matches = sum(face['person_index'] is not None for face in record['faces'])
        self.panel.show(image, f"Faces: {len(record['faces'])}    People: {len(record['people'])}    Matched faces: {matches}")
        self.run_button.state(['!disabled'])
        self.status.set('Detection complete')

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
        try:
            self.media.close(self.browser.close)
        finally:
            self.inference.close(self.detector.close)


def execute(args, config, *, services=None):
    from .run import Services
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError('The detection tuner requires a graphical desktop and working Tcl/Tk. '
                           'Try python -m tkinter in this environment.') from error
    try:
        window = Tuner(root, config, services or Services())
        root.protocol('WM_DELETE_WINDOW', root.quit)
        try:
            if args.scene_id:
                window.scene_entry.insert(0, args.scene_id)
                window.open_scene()
            root.mainloop()
        finally:
            root.withdraw()
            window.close()
    finally:
        root.destroy()
    return 0
