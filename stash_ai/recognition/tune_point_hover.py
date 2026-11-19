"""Native, inference-free tooltips with hit testing in zoomed/scrolled cue coordinates."""
import tkinter as tk
from tkinter import ttk

from .detection_point_hover import nearest


class PointHover:
    def __init__(self, panel):
        self.panel = panel
        self.points = ()
        self.window = self.label = None
        self.selected = ()
        self.text = ''
        panel.canvas.bind('<Motion>', self.motion, add='+')
        panel.canvas.bind('<Leave>', lambda _: self.hide(), add='+')
        panel.canvas.bind('<Destroy>', self.close, add='+')

    def set_points(self, points):
        self.hide()
        self.points = tuple(points)

    def motion(self, event):
        canvas, zoom = self.panel.canvas, self.panel.zoom
        selected = nearest(self.points, canvas.canvasx(event.x)/zoom, canvas.canvasy(event.y)/zoom, zoom)
        if not selected:
            self.hide()
            return
        if selected == self.selected:
            return
        self.selected = selected
        self.text = '\n\n'.join(point.text for point in selected[:3])
        if len(selected) > 3:
            self.text += f'\n\n{len(selected)-3} further overlapping points.'
        if self.window is None:
            self.window = tk.Toplevel(canvas)
            self.window.withdraw()
            self.window.overrideredirect(True)
            self.label = ttk.Label(self.window, padding=8, justify='left', wraplength=440,
                                   relief='solid', borderwidth=1)
            self.label.pack()
        self.label.configure(text=self.text)
        self.window.update_idletasks()
        width, height = self.window.winfo_reqwidth(), self.window.winfo_reqheight()
        x, y = event.x_root+16, event.y_root+16
        if x+width > canvas.winfo_screenwidth():
            x = event.x_root-width-16
        if y+height > canvas.winfo_screenheight():
            y = event.y_root-height-16
        self.window.geometry(f'+{max(0, x)}+{max(0, y)}')
        self.window.deiconify()
        self.window.lift()

    def hide(self):
        self.selected = ()
        self.text = ''
        if self.window is not None:
            self.window.withdraw()

    def close(self, event):
        if event.widget is self.panel.canvas and self.window is not None:
            window = self.window
            self.window = self.label = None
            if window.winfo_exists():
                window.destroy()
