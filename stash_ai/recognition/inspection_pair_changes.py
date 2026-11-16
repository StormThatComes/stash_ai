"""Compare recorded replay decisions and isolate a production stage on demand."""
from functools import partial
import tkinter as tk
from tkinter import ttk

from .inspection_first_difference import first_difference, stage_finding
from .inspection_pair_replay import describe, experiment, partial_state
from .inspection_runtime import Worker
from .replay_intervention import STAGES


FIRST = 'First differing cue'
LATER = 'Later linked changes'
FILTERS = (FIRST, LATER, 'Changed', 'Owner changed', 'A → B', 'B → A', 'Anchor → auxiliary',
           'Unassigned', 'Protected C', 'A residuals', 'B residuals', 'All')


def matches(mode, row, ids, before, after, old_role, new_role):
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


def stage_summary(result):
    stage = STAGES[result['intervention']['stage']]
    lines = [f'{stage} · {stage_finding(result)}']
    for name in ('enabled', 'omitted'):
        trial = result['trials'][name]
        protected = {None: 'unused', True: 'separate', False: 'failed'}[trial['protected']]
        visits = result['intervention'][name + '_visits']
        lines.append(f"{name.capitalize()} · A/B {'joined' if trial['joined'] else 'incomplete'} · "
                     f"C {protected} · {trial['unassigned']} unassigned in scope · "
                     f"{len(trial['outsiders'])} additional identities · {visits} stage entries")
    return '\n'.join(lines)


class ChangesWindow:
    page_size = 100

    def __init__(self, comparison, result, name, before_name='baseline'):
        self.comparison, self.result, self.name = comparison, result, name
        self.before_name = before_name
        self.before, self.after = (result['trials'][n] for n in (before_name, name))
        self.diff = first_difference(result['rows'], self.before, self.after)
        self.labels = dict(zip(result['ids'], 'ABC'))
        self.worker = Worker('pair-stage-necessity')
        self.window = tk.Toplevel(comparison.window)
        self.window.title('Replay observation changes')
        self.window.geometry('1080x720')
        summary = stage_summary(result) if 'intervention' in result else describe(result, name)
        ttk.Label(self.window, text=summary, wraplength=1040, justify='left').pack(anchor='w', padx=6)
        bar = ttk.Frame(self.window, padding=4)
        bar.pack(fill='x')
        self.mode = ttk.Combobox(bar, values=FILTERS, state='readonly', width=22)
        self.mode.set(FIRST if self.diff['first'] else 'All')
        self.mode.pack(side='left')
        self.mode.bind('<<ComboboxSelected>>', self.filter)
        self.previous = ttk.Button(bar, text='Previous', command=lambda: self.show(self.page - 1))
        self.next = ttk.Button(bar, text='Next', command=lambda: self.show(self.page + 1))
        for button in (self.previous, self.next):
            button.pack(side='left')
        evidence = self.after.get('evidence_id')
        if evidence:
            ttk.Button(bar, text='Inspect route evidence', command=lambda: self.open_key(evidence)).pack(side='left')
        self.count = ttk.Label(bar)
        self.count.pack(side='left')
        if self.diff['first']:
            row = result['rows'][self.diff['first'][0]]
            caption = (f"First recorded difference · scene {row['scene']}, cue {row['cue']} · "
                       f"{len(self.diff['first'])} changed here · {len(self.diff['later'])} later changes with a recorded link")
        else:
            caption = 'No recorded observation differences for the selected identities.'
        ttk.Label(self.window, text=caption, padding=4).pack(anchor='w')
        columns = ('cue', 'source', 'old', 'new', 'role', 'route')
        self.tree = ttk.Treeview(self.window, columns=columns, show='headings', selectmode='browse', height=7)
        for col, label, width in zip(columns, ('Scene / cue', 'Source', before_name + ' owner', name + ' owner',
                                                'Role change', 'Recorded decision'), (90,55,140,140,155,280)):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, minwidth=40)
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<Double-1>', self.open)
        self.tree.bind('<Return>', self.open)
        self.tree.bind('<<TreeviewSelect>>', self.details)
        panels = ttk.Frame(self.window)
        panels.pack(fill='x')
        self.panels = []
        for i, title in enumerate((before_name, name)):
            panels.columnconfigure(i, weight=1, uniform='decisions')
            box = ttk.LabelFrame(panels, text=title, padding=5)
            box.grid(row=0, column=i, sticky='nsew')
            label = ttk.Label(box, wraplength=480, justify='left')
            label.pack(anchor='nw')
            self.panels.append(label)
        ttk.Button(self.window, text='Inspect selected cue', command=self.open).pack(pady=3)
        self.patch = result.get('patches', {}).get(name)
        if self.patch is not None and 'intervention' not in result:
            actions = ttk.Frame(self.window, padding=4)
            actions.pack(fill='x')
            self.stage = ttk.Combobox(actions, values=list(STAGES.values()), state='readonly', width=32)
            self.stage.set(STAGES.get(self.patch.get('mechanism'), STAGES['hybrid_merge']))
            self.stage.pack(side='left')
            self.test = ttk.Button(actions, text='Test stage necessity', command=self.test_stage)
            self.test.pack(side='left')
            self.cancel = ttk.Button(actions, text='Cancel', command=self.cancel_test, state='disabled')
            self.cancel.pack(side='left')
            self.status = ttk.Label(actions, wraplength=470)
            self.status.pack(side='left')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.filter()

    def cancel_test(self):
        self.worker.invalidate()
        self.test.state(['!disabled'])
        self.cancel.state(['disabled'])
        self.status.configure(text='Stage check cancelled.')

    def test_stage(self):
        stage = next(key for key, title in STAGES.items() if title == self.stage.get())
        parent = self.comparison
        run = partial(experiment, parent.reader.config, parent.scan['scope'], self.result['ids'],
                      {'enabled': self.patch, 'omitted': self.patch}, parent.revision,
                      scenes=self.result['scenes'], omitted_stage=stage)
        self.test.state(['disabled'])
        self.cancel.state(['!disabled'])
        self.status.configure(text='Comparing the same YAML with the stage enabled and omitted…')
        def failed(error):
            self.test.state(['!disabled'])
            self.cancel.state(['disabled'])
            self.status.configure(text=str(error))
        self.worker.request(run, lambda result: parent.show_changes(result, 'omitted', before_name='enabled'), failed)

    def filter(self, event=None):
        mode = self.mode.get()
        groups = {FIRST: 'first', LATER: 'later', 'Changed': 'changed'}
        if mode in groups:
            self.keys = list(self.diff[groups[mode]])
        else:
            self.keys = []
            for key, row in self.result['rows'].items():
                ids = self.result['ids']
                if mode in ('A residuals', 'B residuals'):
                    src = ids[int(mode == 'B residuals')]
                    keep = row['source'] == src and partial_state(src, self.after['owners'][key], ids, self.after) != 'joined target'
                else:
                    keep = matches(mode, row, ids, self.before['owners'][key], self.after['owners'][key],
                                   self.before['decisions'].get(key, {}).get('role'),
                                   self.after['decisions'].get(key, {}).get('role'))
                if keep:
                    self.keys.append(key)
        rows = self.result['rows']
        self.keys.sort(key=lambda k: (int(rows[k]['scene']), rows[k]['cue'], k))
        self.show(0)

    def owner(self, value):
        return self.labels.get(value, value) or 'Unassigned'

    def show(self, page):
        self.page = page
        self.tree.delete(*self.tree.get_children())
        keys = self.keys[page*self.page_size:(page+1)*self.page_size]
        for key in keys:
            row = self.result['rows'][key]
            old, new = (trial['decisions'].get(key, {}) for trial in (self.before, self.after))
            self.tree.insert('', 'end', iid=key, values=(
                f"{row['scene']} / {row['cue']}", self.labels[row['source']], self.owner(self.before['owners'][key]),
                self.owner(self.after['owners'][key]), f"{old.get('role','—')} → {new.get('role','—')}",
                ' · '.join(filter(None, (new.get('route'), new.get('stage'), *(e[2] for e in new.get('links', ()))))) or 'Unavailable'))
        self.previous.state(['!disabled'] if page else ['disabled'])
        self.next.state(['!disabled'] if (page+1)*self.page_size < len(self.keys) else ['disabled'])
        self.count.configure(text=f'{len(self.keys)} observations · page {page+1}')
        if keys:
            self.tree.selection_set(keys[0])
        self.details()

    def details(self, event=None):
        selected = self.tree.selection()
        for panel, trial in zip(self.panels, (self.before, self.after)):
            lines = []
            if selected:
                key = selected[0]
                d = trial['decisions'].get(key, {})
                lines.append(f"Owner {self.owner(trial['owners'][key])} · {d.get('role', 'Role unavailable')}")
                lines.append(' · '.join(filter(None, (d.get('stage'), d.get('route'), d.get('signal')))) or 'Decision unavailable')
                for source, target, route in d.get('links', ()):
                    left = self.owner(source) if source else 'Unknown source'
                    right = self.owner(target) if target else 'Unknown destination'
                    lines.append(f'{left} → {right} · {route or "identity merge"}')
                refs = d.get('references', ())
                if refs:
                    rows = self.result['rows']
                    labels = [f"{rows[r]['kind']} {rows[r]['scene']}/{rows[r]['cue']}" for r in refs[:6]]
                    lines.append('Recorded references: ' + ', '.join(labels) + (f' (+{len(refs)-6})' if len(refs)>6 else ''))
            panel.configure(text='\n'.join(lines))

    def open(self, event=None):
        selection = self.tree.selection()
        if selection:
            self.open_key(selection[0])

    def open_key(self, key):
        row = self.result['rows'][key]
        self.comparison.view(dict(label=f"Replay observation · {row['id']}", refs=[(row['kind'], row['id'])]))

    def close(self):
        self.worker.close()
        self.window.destroy()
        self.comparison.changes_window = None
