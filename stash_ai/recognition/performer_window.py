"""Performer entry point sharing observation previews with the scene inspector."""
from functools import partial
import tkinter as tk
from tkinter import ttk, simpledialog
from .identity_window import Inspector
from .identity_inspection import IdentityReader
from .inspection_runtime import Worker
from .performer_inspection import summary


class PerformerInspector(Inspector):
    def build_browser(self, panel, observation):
        super().build_browser(panel, observation)
        self.metadata_worker = Worker('performer-summary')
        self.summary_data = None
        self.grouping = tk.StringVar(master=self.root, value='Scene')
        bar = self.entries['scene'].master
        ttk.Button(bar, text='Performer…', command=self.change_performer).pack(side='left')
        ttk.Button(bar, text='Inspect scene', command=self.open_scene).pack(side='left')
        self.kind = ttk.Combobox(bar, values=['All', 'face', 'body'], width=6, state='readonly')
        self.kind.set('All'); self.kind.pack(side='left')
        self.kind.bind('<<ComboboxSelected>>', lambda _: self.search())
        self.state = ttk.Combobox(bar, values=['All','anchor','auxiliary','attachment','unknown'], width=10, state='readonly')
        self.state.set('All'); self.state.pack(side='left')
        self.state.bind('<<ComboboxSelected>>', lambda _: self.search())
        group = ttk.Combobox(bar, textvariable=self.grouping, values=['Scene','Role'], width=6, state='readonly')
        group.pack(side='left'); group.bind('<<ComboboxSelected>>', lambda _: self.page(0))
        self.listing.configure(show='tree headings', columns=('scene','cue','kind','state'))
        self.listing.heading('#0', text='Group / observation'); self.listing.column('#0', width=180)
        self.listing.heading('state', text='Role'); self.listing.column('state', width=85)
        self.summary_label = ttk.Label(self.listing.master, wraplength=410)
        self.summary_label.grid(row=2, column=0, columnspan=2, sticky='w')

    def search(self):
        super().search()
        self.metadata_worker.request(partial(summary, self.reader), self.show_summary, self.error)

    def page(self, direction):
        self.filters.update(kind=None if self.kind.get()=='All' else self.kind.get(),
                            state=None if self.state.get()=='All' else self.state.get(), accepted_only=True)
        super().page(direction)

    def show_summary(self, result):
        self.summary_data = result
        self.root.title('Performer inspector · ' + (result['person']['performer'] or result['person']['id']))
        counts = {}
        for g in result['groups']:
            key = g['kind'] + ' ' + g['state']
            counts[key] = counts.get(key, 0) + g['count']
        self.summary_label.configure(text=f"{result['scenes']} scenes · {result['observations']} observations\n" +
                                     ' · '.join(f'{k}: {v}' for k,v in sorted(counts.items())) +
                                     '\nGroups below cover this page; filters search all current members.')

    def show_page(self, result):
        rows, more = result
        groups = {}
        for row in rows:
            key = row['scene'] if self.grouping.get()=='Scene' else row['state']
            if key not in groups:
                groups[key] = self.listing.insert('', 'end', text=key, open=True)
            item = self.listing.insert(groups[key], 'end', text=row['id'],
                values=(row['scene'], row['cue'], row['kind'], row['state']))
            self.rows[item] = row
        self.prev.state(['!disabled' if self.offset else 'disabled'])
        self.next.state(['!disabled' if more else 'disabled'])
        if self.rows:
            self.listing.selection_set(next(iter(self.rows)))
        else:
            self.status.set('No current observations match these filters.')

    def select(self, event=None):
        selected = self.listing.selection()
        if selected and selected[0] in self.rows:
            self.detail_reader = self.rows[selected[0]].get('reader', self.reader)
        super().select(event)

    def change_performer(self):
        value = simpledialog.askstring('Inspect performer', 'Numeric performer ID', parent=self.root)
        if value and value.isascii() and value.isdecimal() and int(value)>0:
            self.reader = IdentityReader(self.reader.config, performer_id=str(int(value)), scope=self.reader.selector[3])
            self.search()

    def open_scene(self):
        if not self.current_row:
            self.status.set('Select an observation first.'); return
        from .scene_window import SceneInspector
        from .scene_inspection import SceneReader
        open_window(self, SceneInspector, SceneReader(self.reader.config, self.current_row['scene'], scope=self.preview_scope))

    def poll(self):
        self.metadata_worker.poll()
        super().poll()

    def close(self):
        self.metadata_worker.close()
        super().close()


def open_window(host, cls, reader):
    """Child inspectors own their workers and previews, with an idempotent close."""
    root = tk.Toplevel(host.root)
    ui = cls(root, reader, '', previews=host.previews.fork() if host.previews else None)
    class Child:
        closed = False
        def close(self):
            if self.closed: return
            self.closed = True
            ui.close(); root.destroy()
            if self in host.visual_windows: host.visual_windows.remove(self)
    child = Child()
    host.visual_windows.append(child)
    root.protocol('WM_DELETE_WINDOW', child.close)
    return ui
