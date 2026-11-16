"""On-demand comparison of two identities from the scene catalogue."""
from functools import partial
import tkinter as tk
from tkinter import ttk, simpledialog
from tkinter.scrolledtext import ScrolledText

from .inspection_merge_paths import analyse
from .inspection_pair_replay import describe, experiment
from .inspection_runtime import Worker


class MergeWindow:
    def __init__(self, host, groups):
        self.host, self.reader, self.scan = host, host.reader, host.scan
        self.ids = [g['identity'] for g in groups]
        self.witness_window = self.changes_window = self.variants_window = None
        self.fragment_window = None
        self.views_window = None
        self.matrix, self.hybrid = {}, {}
        self.protected = None
        self.results = {}
        self.labels = [(f"Performer {g['performer']}" if g['performer'] else 'Unpublished identity') + ' · ' + g['identity']
                  for g in groups]
        self.window = tk.Toplevel(host.root)
        self.window.title('Identity merge paths · Scene '+self.scan['scene'])
        self.window.geometry('1020x640')
        self.worker = Worker('scene-merge-paths')
        self.items, self.replays = {}, {}
        self.busy = False
        self.revision = None
        self.selectors = []
        bar = ttk.Frame(self.window, padding=6)
        bar.pack(fill='x')
        for index, label in enumerate(('Identity A','Identity B')):
            ttk.Label(bar, text=label).grid(row=index,column=0,sticky='w')
            combo = ttk.Combobox(bar, values=self.labels, state='readonly', width=54)
            combo.grid(row=index,column=1,sticky='ew')
            if index < len(groups):
                combo.current(index)
            combo.bind('<<ComboboxSelected>>', self.reset)
            self.selectors.append(combo)
        ttk.Button(bar, text='Find B by ID…', command=self.find_b).grid(row=1,column=3)
        self.protect_label = ttk.Label(bar, text='Protected C (optional): none')
        self.protect_label.grid(row=2,column=0,columnspan=2,sticky='w')
        ttk.Button(bar,text='Choose C…',command=self.find_c).grid(row=2,column=2)
        ttk.Button(bar,text='Clear C',command=lambda:self.set_c(None)).grid(row=2,column=3)
        bar.columnconfigure(1, weight=1)
        self.run = ttk.Button(bar, text='Analyse paths', command=self.load)
        self.run.grid(row=0,column=2,padx=5)
        self.cancel_button = ttk.Button(bar, text='Cancel', command=self.reset)
        self.cancel_button.grid(row=1,column=2)
        self.status = ttk.Label(self.window, wraplength=940, padding=6)
        self.status.pack(fill='x')
        split = ttk.Panedwindow(self.window, orient='horizontal')
        split.pack(fill='both', expand=True, padx=6)
        left = ttk.Frame(split)
        split.add(left, weight=1)
        self.tree = ttk.Treeview(left, columns=('status',), selectmode='browse')
        self.tree.heading('#0', text='Route / YAML option')
        self.tree.heading('status', text='Evidence')
        self.tree.column('#0', width=285)
        self.tree.column('status', width=155)
        scroll = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y')
        self.tree.pack(fill='both',expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.select)
        self.detail = ScrolledText(split, width=48, wrap='word', state='disabled', padx=8, pady=8)
        split.add(self.detail, weight=1)
        evidence = ttk.Frame(self.window, padding=4)
        evidence.pack(fill='x')
        self.witnesses = ttk.Combobox(evidence, state='readonly', width=45)
        self.witnesses.pack(side='left')
        self.open_evidence = ttk.Button(evidence, text='Inspect evidence', command=self.open_witness)
        self.open_evidence.pack(side='left')
        self.views = ttk.Button(evidence,text='View / Hybrid evidence…',command=self.open_views)
        self.views.pack(side='left',padx=4)
        self.fragments = ttk.Button(evidence,text='Attachment scenarios…',command=self.open_fragments)
        self.fragments.pack(side='left')
        actions = ttk.Frame(self.window, padding=6)
        actions.pack(fill='x')
        self.copy = ttk.Button(actions, text='Copy YAML patch', command=self.copy_yaml)
        self.copy.pack(side='left')
        self.replay = ttk.Button(actions, text='Verify option by replay', command=self.verify)
        self.replay.pack(side='left', padx=6)
        self.changes = ttk.Button(actions,text='Observation changes…',command=self.open_changes)
        self.changes.pack(side='left')
        self.batch = ttk.Button(actions,text='Compare YAML variants…',command=self.open_variants)
        self.batch.pack(side='left',padx=4)
        self.reset()
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = self.window.after(50, self.poll)

    def pair(self):
        return tuple(self.ids[c.current()] for c in self.selectors if c.current() >= 0)

    def find_b(self):
        from .inspection_pair_evidence import find_identity
        value = simpledialog.askstring('Find identity B', 'Stash performer ID or identity UUID in this scope:', parent=self.window)
        if value and value.strip():
            self.reset()
            self.start('Finding current identity…', partial(find_identity, self.reader, self.scan['scope'], value.strip()), self.set_b)

    def open_fragments(self):
        from .inspection_fragment_window import FragmentWindow
        if self.fragment_window:
            self.fragment_window.window.lift()
        else:
            self.fragment_window = FragmentWindow(self)

    def open_views(self):
        from .inspection_views_window import ViewsWindow
        if self.views_window:
            self.views_window.window.lift()
        else:
            self.views_window = ViewsWindow(self)

    def open_variants(self):
        from .inspection_variants_window import VariantsWindow
        if self.variants_window:
            self.variants_window.window.lift()
        else:
            self.variants_window = VariantsWindow(self)

    def find_c(self):
        from .inspection_pair_evidence import find_identity
        value = simpledialog.askstring('Protect identity C', 'Stash performer ID or identity UUID:', parent=self.window)
        if value and value.strip():
            self.start('Finding protected identity…', partial(find_identity, self.reader, self.scan['scope'], value.strip()),
                       lambda group:self.set_c(group['identity']))

    def set_c(self, identity):
        if identity is not None and identity in self.pair():
            self.ready('C must be different from both A and B.')
            return
        self.protected = identity
        self.protect_label.configure(text='Protected C (optional): ' + (identity or 'none'))
        self.reset()

    def open_changes(self):
        selected = self.tree.selection()
        if selected and selected[0] in self.results:
            result, name = self.results[selected[0]]
            self.show_changes(result, name)

    def show_changes(self, result, name, before_name="baseline"):
        from .inspection_pair_changes import ChangesWindow
        if self.changes_window:
            self.changes_window.close()
        self.changes_window = ChangesWindow(self, result, name, before_name)

    def set_b(self, group):
        address = group['identity']
        if address not in self.ids:
            self.ids.append(address)
            self.labels.append((f"Performer {group['performer']}" if group['performer'] else 'Unpublished identity') + ' · ' + address)
            self.selectors[1].configure(values=self.labels)
        self.selectors[1].current(self.ids.index(address))
        self.reset()

    def open_witness(self):
        index = self.witnesses.current()
        if index >= 0:
            self.view(self.refs[index])

    def view(self, evidence):
        from .inspection_witness_window import WitnessWindow
        if self.witness_window:
            self.witness_window.close()
        self.witness_window = WitnessWindow(self, evidence)

    def reset(self, event=None):
        if self.fragment_window:
            self.fragment_window.close()
        if self.views_window:
            self.views_window.close()
        if self.witness_window:
            self.witness_window.close()
        if self.changes_window:
            self.changes_window.close()
        if self.variants_window:
            self.variants_window.close()
        self.results.clear()
        self.worker.invalidate()
        self.items.clear()
        self.replays.clear()
        self.revision = None
        self.tree.delete(*self.tree.get_children())
        valid = len(set(self.pair())) == 2 and self.protected not in self.pair()
        self.ready('Select Analyse paths to compare this pair.' if valid else 'A, B and optional C must be different identities.')

    def ready(self, message):
        self.busy = False
        self.run.state(['!disabled'] if len(set(self.pair())) == 2 and self.protected not in self.pair() else ['disabled'])
        self.cancel_button.state(['disabled'])
        self.status.configure(text=message)
        self.select()

    def start(self, message, operation, complete):
        self.busy = True
        self.run.state(['disabled'])
        self.cancel_button.state(['!disabled'])
        self.select()
        self.status.configure(text=message)
        self.worker.request(operation, complete, lambda error: self.ready(str(error)))

    def load(self):
        self.reset()
        if len(set(self.pair())) == 2 and self.protected not in self.pair():
            self.start('Comparing retained evidence and checking catalogue proposals…',
                       partial(analyse, self.reader, self.scan, self.pair()), self.show)

    def show(self, result):
        self.matrix, self.hybrid = result.get('matrix',{}),result.get('hybrid',{})
        self.revision = result.get('revision')
        for route in result['routes']:
            parent = self.tree.insert('', 'end', text=route['title'], values=(route['status'],), open=True)
            self.items[parent] = (route, None)
            for patch in route['patches']:
                item = self.tree.insert(parent, 'end', text=patch['title'].capitalize(), values=('YAML option' if patch['changes'] else 'Current YAML',))
                self.items[item] = (route, patch)
        if self.items:
            self.tree.selection_set(next(iter(self.items)))
        self.ready(result['summary'])

    def select(self, event=None):
        selected = self.tree.selection()
        item = selected[0] if selected else None
        route, patch = self.items.get(item, (None,None))
        lines = [route['title'], route['status'], '', *route['notes']] if route else []
        self.refs = (patch or {}).get('witnesses', []) + (route or {}).get('witnesses', [])
        self.witnesses.configure(values=[ref['label'] for ref in self.refs])
        self.witnesses.set('')
        if self.refs:
            self.witnesses.current(0)
        self.open_evidence.state(['!disabled'] if self.refs else ['disabled'])
        if route:
            lines.extend(('', 'First blocker: ' + (route.get('first_blocker') or 'none recorded')))
            for step in route.get('steps', ()):
                lines.append(f"{step['status']} · {step['label']}  {step['detail']}")
        if patch is not None:
            lines.extend(('', patch['title'], *patch.get('evidence', ()), *patch.get('notes', ())))
            for change in patch['changes']:
                key = change.get('yaml_parameter',change['parameter'])
                lines.append(f"{key}: {change['current']} → {change['proposed']}")
            lines.extend(('', patch['yaml'].rstrip() or 'Use the current YAML.'))
        elif route and route['patches']:
            lines.extend(('', 'Select a YAML option below this route to inspect its changes or verify it by replay.'))
        if item in self.replays:
            lines.extend(('', self.replays[item]))
        self.detail.configure(state='normal')
        self.detail.delete('1.0','end')
        self.detail.insert('1.0','\n'.join(lines))
        self.detail.configure(state='disabled')
        self.fragments.state(['!disabled'] if self.revision else ['disabled'])
        self.views.state(['!disabled'] if self.revision else ['disabled'])
        self.batch.state(['!disabled'] if self.revision and any(p is not None for _,p in self.items.values()) and not self.busy else ['disabled'])
        self.changes.state(['!disabled'] if item in self.results else ['disabled'])
        self.copy.state(['!disabled'] if patch and patch['yaml'] else ['disabled'])
        self.replay.state(['!disabled'] if patch is not None and not self.busy else ['disabled'])

    def copy_yaml(self):
        _, patch = self.items[self.tree.selection()[0]]
        self.window.clipboard_clear()
        self.window.clipboard_append(patch['yaml'])

    def verify(self):
        item = self.tree.selection()[0]
        _, patch = self.items[item]
        ids = self.pair() + ((self.protected,) if self.protected else ())
        def complete(result):
            self.results[item] = (result, 'option')
            self.replays[item] = describe(result, 'option')
            self.ready('Replay finished. Select the verified option to see its result.')
        self.start('Replaying the current YAML and selected option on retained scans…',
                   partial(experiment, self.reader.config, self.scan['scope'], ids, {'option':patch}, self.revision), complete)

    def poll(self):
        self.worker.poll()
        if self.changes_window:
            self.changes_window.worker.poll()
        if self.fragment_window:
            self.fragment_window.worker.poll()
        if self.views_window:
            self.views_window.worker.poll()
        if self.variants_window:
            self.variants_window.poll()
        if self.witness_window:
            self.witness_window.worker.poll()
        self.timer = self.window.after(50, self.poll)

    def close(self):
        self.window.after_cancel(self.timer)
        if self.fragment_window:
            self.fragment_window.close()
        if self.views_window:
            self.views_window.close()
        if self.witness_window:
            self.witness_window.close()
        if self.changes_window:
            self.changes_window.close()
        if self.variants_window:
            self.variants_window.close()
        self.worker.close()
        self.window.destroy()
        self.host.visual_windows.remove(self)
        self.host.merge_window = None
