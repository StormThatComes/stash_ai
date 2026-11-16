"""Paged, filterable observation changes for one pair replay."""
import tkinter as tk
from tkinter import ttk

from .inspection_pair_replay import describe


FILTERS = ('Changed', 'Owner changed', 'A → B', 'B → A', 'Anchor → auxiliary', 'Unassigned', 'Protected C', 'All')


def matches(mode, row, ids, before, after, old_role, new_role):
    if mode == 'Changed':
        return before != after or old_role != new_role
    if mode == 'Owner changed':
        return before != after
    if mode in ('A → B', 'B → A'):
        source, target = ids[:2] if mode == 'A → B' else ids[:2][::-1]
        return row['source'] == source and after == target and before != after
    if mode == 'Anchor → auxiliary':
        return old_role == 'anchor' and new_role == 'auxiliary'
    if mode == 'Unassigned':
        return after is None
    if mode == 'Protected C':
        return len(ids) == 3 and row['source'] == ids[2]
    return True


class ChangesWindow:
    page_size = 100

    def __init__(self, comparison, result, name):
        self.comparison, self.result, self.name = comparison, result, name
        self.window = tk.Toplevel(comparison.window)
        self.window.title('Replay observation changes')
        self.window.geometry('1060x610')
        ttk.Label(self.window, text=describe(result,name), wraplength=1000).pack(anchor='w', padx=6)
        bar = ttk.Frame(self.window, padding=4)
        bar.pack(fill='x')
        self.mode = ttk.Combobox(bar, values=FILTERS, state='readonly', width=20)
        self.mode.current(0)
        self.mode.pack(side='left')
        self.mode.bind('<<ComboboxSelected>>', self.filter)
        self.previous = ttk.Button(bar, text='Previous', command=lambda:self.show(self.page-1))
        self.next = ttk.Button(bar, text='Next', command=lambda:self.show(self.page+1))
        for button in (self.previous,self.next):
            button.pack(side='left')
        self.count = ttk.Label(bar)
        self.count.pack(side='left')
        columns = ('cue', 'source', 'old', 'new', 'role', 'route')
        self.tree = ttk.Treeview(self.window, columns=columns, show='headings', selectmode='browse')
        for col, label, width in zip(columns, ('Scene / cue', 'Source', 'Baseline owner', 'Variant owner', 'Baseline → variant role', 'Recorded decision'),
                                     (90,55,165,165,155,260)):
            self.tree.heading(col,text=label)
            self.tree.column(col,width=width,minwidth=40)
        self.tree.pack(fill='both',expand=True)
        self.tree.bind('<Double-1>',self.open)
        self.tree.bind('<Return>',self.open)
        ttk.Button(self.window,text='Inspect selected cue',command=self.open).pack(pady=3)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.filter()

    def filter(self, event=None):
        self.keys = []
        result = self.result
        before, after = result['trials']['baseline'], result['trials'][self.name]
        for key,row in result['rows'].items():
            if matches(self.mode.get(), row, result['ids'], before['owners'][key], after['owners'][key],
                       before['decisions'].get(key,{}).get('role'), after['decisions'].get(key,{}).get('role')):
                self.keys.append(key)
        self.keys.sort(key=lambda k:(int(result['rows'][k]['scene']),result['rows'][k]['cue'],k))
        self.show(0)

    def show(self, page):
        self.page = page
        result = self.result
        labels = dict(zip(result['ids'],'ABC'))
        def owner(value):
            return labels.get(value,value) or 'Unassigned'
        before, after = result['trials']['baseline'], result['trials'][self.name]
        self.tree.delete(*self.tree.get_children())
        for key in self.keys[page*self.page_size:(page+1)*self.page_size]:
            row = result['rows'][key]
            old, new = before['decisions'].get(key,{}), after['decisions'].get(key,{})
            self.tree.insert('', 'end', iid=key, values=(
                f"{row['scene']} / {row['cue']}", labels[row['source']], owner(before['owners'][key]),
                owner(after['owners'][key]), f"{old.get('role','—')} → {new.get('role','—')}",
                ' · '.join(filter(None, (new.get('route'),new.get('stage'),*(e[2] for e in new.get('links',()))))) or 'Unavailable'))
        self.previous.state(['!disabled'] if page else ['disabled'])
        self.next.state(['!disabled'] if (page+1)*self.page_size<len(self.keys) else ['disabled'])
        self.count.configure(text=f'{len(self.keys)} observations · page {page+1}')

    def open(self, event=None):
        selection = self.tree.selection()
        if selection:
            row = self.result['rows'][selection[0]]
            self.comparison.view(dict(label=f"Replay observation · {row['id']}",refs=[(row['kind'],row['id'])]))

    def close(self):
        self.window.destroy()
        self.comparison.changes_window = None
