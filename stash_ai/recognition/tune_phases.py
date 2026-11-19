"""A second, inference-free navigation bar for the selected A/B panel."""
import tkinter as tk
from tkinter import ttk


class PhaseNavigation(ttk.Frame):
    def __init__(self, parent, images):
        super().__init__(parent)
        self.images = images
        self.results = {}
        self.positions = {}
        self.panel = 'A'
        self.updating = False
        self.caption = tk.StringVar(value='Phase A: run detection')
        ttk.Label(self, textvariable=self.caption, wraplength=900).grid(row=0, column=0, columnspan=4, sticky='w')
        self.previous = ttk.Button(self, text='Previous phase', command=lambda: self.step(-1))
        self.previous.grid(row=1, column=0)
        self.slider = ttk.Scale(self, from_=0, to=0, command=self.slide, state='disabled')
        self.slider.grid(row=1, column=1, sticky='ew', padx=8)
        self.next = ttk.Button(self, text='Next phase', command=lambda: self.step(1))
        self.next.grid(row=1, column=2)
        self.final = ttk.Button(self, text='Final result', command=self.show_final)
        self.final.grid(row=1, column=3, padx=(8, 0))
        self.columnconfigure(1, weight=1)
        self.slider.bind('<Left>', lambda _: self.step(-1))
        self.slider.bind('<Right>', lambda _: self.step(1))
        self.refresh()

    def select_panel(self, name):
        self.panel = name
        self.refresh()

    def clear(self, name):
        self.results.pop(name, None)
        self.positions.pop(name, None)
        self.refresh()

    def set_result(self, name, result, source):
        history = getattr(result, 'phases', None)
        self.results[name] = result, source, history
        self.positions[name] = len(history.phases) if history is not None else 0
        self.show(name)
        self.refresh()

    def refresh(self):
        self.updating = True
        try:
            item = self.results.get(self.panel)
            last = len(item[2].phases) if item is not None and item[2] is not None else 0
            position = self.positions.get(self.panel, 0)
            self.slider.configure(to=last, state='normal' if last else 'disabled')
            self.slider.set(position)
            for button, enabled in ((self.previous, position > 0), (self.next, position < last),
                                    (self.final, position < last)):
                button.state(['!disabled' if enabled else 'disabled'])
            if item is None:
                self.caption.set(f'Phase {self.panel}: run detection')
            else:
                title = 'Final result' if position == last else item[2].phases[position].title
                suffix = f' | history limit: {item[2].omitted} phases omitted' if item[2] and item[2].omitted else ''
                self.caption.set(f'Phase {self.panel} {position+1}/{last+1}: {title}{suffix}')
        finally:
            self.updating = False

    def slide(self, value):
        if self.updating or self.panel not in self.results:
            return
        history = self.results[self.panel][2]
        last = len(history.phases) if history is not None else 0
        position = min(max(round(float(value)), 0), last)
        if position != self.positions[self.panel]:
            self.positions[self.panel] = position
            self.show(self.panel)
            self.refresh()

    def step(self, amount):
        self.slide(self.positions.get(self.panel, 0) + amount)
        return 'break'

    def show_final(self):
        item = self.results.get(self.panel)
        if item is not None:
            self.slide(len(item[2].phases) if item[2] is not None else 0)

    def show(self, name):
        from .detection_point_hover import targets, face_targets, draw_rejected
        from .detection_point_details import final_people
        result, source, history = self.results[name]
        position = self.positions[name]
        if history is not None and position < len(history.phases):
            image = history.render(source, position)
            caption = f'Phase {position+1}/{len(history.phases)+1}: {history.phases[position].title}'
            phase = history.phases[position]
            points = targets(phase.people, phase.title, source.size)
        else:
            image, record = result
            people = final_people(record, history)
            points = targets(people, 'Final result', source.size) + face_targets(record)
            if any(point.rejected for point in points):
                image = image.copy()
                draw_rejected(image, people, source.size)
            matches = sum(face['person_index'] is not None for face in record['faces'])
            caption = (f"Final result | Faces: {len(record['faces'])}    People: {len(record['people'])}"
                       f'    Matched faces: {matches}')
        self.images.show(name, image, caption, points=points)
