"""Side-by-side audit panels with one source-pixel scale and scroll position."""
import math
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


class ImagePanel(ttk.LabelFrame):
    def __init__(self, parent, title, pan):
        super().__init__(parent, text=title, padding=5)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, background='#20242a', highlightthickness=0, width=260, height=220)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.item = self.canvas.create_image(0, 0, anchor='nw')
        self.image = self.photo = None
        self.zoom = 1.
        self.redraw = None
        horizontal = ttk.Scrollbar(self, orient='horizontal', command=lambda *args: pan('x', *args))
        vertical = ttk.Scrollbar(self, orient='vertical', command=lambda *args: pan('y', *args))
        horizontal.grid(row=1, column=0, sticky='ew')
        vertical.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind('<Configure>', lambda _: self.refresh())
        self.caption = tk.StringVar()
        ttk.Label(self, textvariable=self.caption, wraplength=400).grid(row=2, column=0, sticky='w')

    def refresh(self):
        if self.redraw is None:
            self.redraw = self.after_idle(self.draw)

    def draw(self):
        self.redraw = None
        if self.image is None:
            self.canvas.itemconfigure(self.item, image='')
            self.photo = None
            return
        # Render only visible pixels, even at 400% with extended SAM panels.
        x = max(0, int(self.canvas.canvasx(0)))
        y = max(0, int(self.canvas.canvasy(0)))
        right = min(x + self.canvas.winfo_width(), math.ceil(self.image.width * self.zoom))
        bottom = min(y + self.canvas.winfo_height(), math.ceil(self.image.height * self.zoom))
        if right <= x or bottom <= y:
            self.canvas.itemconfigure(self.item, image='')
            self.photo = None
            return
        box = (x / self.zoom, y / self.zoom, min(self.image.width, right / self.zoom),
               min(self.image.height, bottom / self.zoom))
        with self.image.resize((right-x, bottom-y), Image.Resampling.NEAREST, box=box) as visible:
            self.photo = ImageTk.PhotoImage(visible, master=self.canvas)
        self.canvas.coords(self.item, x, y)
        self.canvas.itemconfigure(self.item, image=self.photo)


class ImagePair(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.panels = {name: ImagePanel(self, 'Configuration ' + name, self.pan) for name in ('A', 'B')}
        self.zoom = 1.
        self.compare(False)

    def compare(self, enabled):
        self.names = ('A', 'B') if enabled else ('A',)
        for column, (name, panel) in enumerate(self.panels.items()):
            visible = name in self.names
            self.columnconfigure(column, weight=int(visible), uniform='panel' if visible else '')
            if visible:
                panel.grid(row=0, column=column, sticky='nsew', padx=2)
            else:
                panel.grid_remove()
        self.layout()

    def show(self, name, image, caption):
        panel = self.panels[name]
        panel.image = image
        panel.caption.set(caption)
        self.layout()

    def layout(self):
        images = [self.panels[name].image for name in self.names if self.panels[name].image is not None]
        extent = tuple(math.ceil(max((im.size[i] for im in images), default=0) * self.zoom)
                       for i in (0, 1))
        for name in self.names:
            panel = self.panels[name]
            panel.zoom = self.zoom
            panel.canvas.configure(scrollregion=(0, 0, *extent))
            panel.refresh()
        self.align()

    def align(self):
        if len(self.names) == 2:
            a, b = self.panels['A'].canvas, self.panels['B'].canvas
            b.xview_moveto(a.xview()[0])
            b.yview_moveto(a.yview()[0])
            self.panels['B'].refresh()

    def pan(self, axis, *args):
        panel = self.panels['A']
        getattr(panel.canvas, axis + 'view')(*args)
        panel.refresh()
        self.align()
