"""Native, read-only desktop inspector. All database work runs off the Tk thread."""
import gc
import json
import tkinter as tk
from tkinter import ttk
from functools import partial

from .identity_inspection import IdentityReader, PAGE_SIZE
from .tune_images import ImagePair
from .tune_worker import Worker


def describe(trace):
    lines = [f"{trace['status'].capitalize()} · {trace['role']} · {trace['evidence_status'].replace('_', ' ')}"]
    if not trace['decisions']:
        lines.append('No accepted identity membership is recorded.' if trace['status']=='unassigned' else
                     'Membership is recorded. No accepted decision explaining it was retained.')
    for decision in trace['decisions']:
        lines.extend(('', decision['summary'], f"Reason: {decision['reason']}"))
        if decision.get('reference_role'):
            lines.append('Recorded reference role: ' + decision['reference_role'])
        metrics = decision['metrics']
        for key, value in metrics.items():
            label = key.replace('_', ' ').capitalize()
            lines.append(f'{label}: {value:.4f}' if isinstance(value, float) else f'{label}: {value}')
        score = metrics.get('score', metrics.get('match_score', metrics.get('mean_score', metrics.get('similarity'))))
        runner = metrics.get('runner', metrics.get('rival_score'))
        if score is not None and runner is not None:
            lines.append(f"Observed lead: {score - runner:.4f}")
        if not decision['references']:
            lines.append('No source observation IDs retained for this decision.')
        scores = metrics.get('recorded_scores') or ()
        if not any(k.rsplit('.',1)[-1] in ('runner','rival','rival_score','rival_cluster') for k in metrics) and len(scores)<2:
            lines.append('No competitor result retained.')
        for edge in decision['merges']:
            route = edge.get('route') or 'recorded identity merge'
            lines.append(f"Merge: {edge['source'] or 'unknown'} → {edge['target'] or 'replay survivor'} ({route})")
        if decision['merge_source'] and not decision['merges']:
            lines.append(f"Recorded predecessor: {decision['merge_source']} (merge details unavailable)")
        lines.append(f"Matching policy: {decision['policy']}")
    return '\n'.join(lines)


def describe_diagnostics(result):
    if 'admission' not in result:
        return ''
    from .admission_panel import SOURCES
    lines = ['', 'Retained rejection: ' + (result['rejection'] or 'none'),
             'Scan status: ' + result['scan_status'],
             'Individual quality gates do not establish identity membership.']
    if result.get('contract_current') is False:
        lines.append('Historical body collection contract differs from the current YAML.')
    for decision in result.get('recorded_decisions', []):
        lines.extend(('', 'Recorded ' + decision['status'] + ': ' + str(decision['recorded'].get('reason', 'unknown')),
                      'Policy: ' + decision['policy'] + (' (current)' if decision['current'] else ' (other policy)')))
        if decision.get('person'):
            lines.append('Recorded candidate identity: ' + decision['person'])
        lines.append(decision['explanation'])
    for group in result['admission']['groups']:
        lines.extend(('', group['stage'] + ' · ' + group['status'].upper(), SOURCES[group['configuration_source']]))
        for check in group['checks']:
            value = check['measured'] if check['measured'] is not None else 'unknown'
            lines.append(f"[{check['status'].upper()}] {check['parameter']}: {value} {check['operator']} {check['threshold']}")
        if group.get('note'):
            lines.append(group['note'])
    return '\n' + '\n'.join(lines)


class Inspector:
    def __init__(self, root, reader, title, observation=None, *, previews=None):
        self.root, self.reader = root, reader
        self.previews, self.preview_scope = previews, None
        self.detail_reader = reader
        self.worker = Worker('identity-inspector')
        self.offset = 0
        self.refs, self.proofs = [], []
        self.filters = {}
        root.title('Identity inspector · ' + title)
        root.geometry('1080x720')
        panel = self.panel = ttk.Frame(root, padding=10)
        panel.pack(fill='both', expand=True)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(1, weight=1)
        self.build_browser(panel, observation)
        right = ttk.Frame(panel)
        right.grid(row=1, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(2, weight=1)
        self.images = ImagePair(right)
        self.images.panels['A'].configure(text='Selected observation')
        self.images.panels['B'].configure(text='Recorded reference')
        self.images.compare(True)
        self.images.grid(row=0, column=0, sticky='nsew')
        refbar = ttk.Frame(right)
        refbar.grid(row=1, column=0, sticky='ew', pady=6)
        ttk.Label(refbar, text='Reference').pack(side='left')
        self.references = ttk.Combobox(refbar, state='readonly')
        self.references.pack(side='left', fill='x', expand=True, padx=4)
        self.references.bind('<<ComboboxSelected>>', self.show_reference)
        self.proof_button = ttk.Button(refbar, text='Recorded evidence', command=self.show_proofs)
        self.proof_button.pack(side='right')
        self.notes = tk.Text(right, height=10, width=40, wrap='word', state='disabled')
        self.notes.grid(row=2, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(right, command=self.notes.yview)
        scrollbar.grid(row=2, column=1, sticky='ns')
        self.notes.configure(yscrollcommand=scrollbar.set)
        self.status = tk.StringVar()
        ttk.Label(panel, textvariable=self.status).grid(row=2, column=0, columnspan=2, sticky='w')
        self.rows = {}
        self.poll_id = root.after(30, self.poll)
        self.search()

    def build_browser(self, panel, observation):
        bar = ttk.Frame(panel)
        bar.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0,8))
        self.entries = {}
        for key, label, width in (('scene','Scene ID',9), ('cue','Cue',7), ('observation','Observation ID',32)):
            ttk.Label(bar, text=label).pack(side='left')
            entry = self.entries[key] = ttk.Entry(bar, width=width)
            entry.pack(side='left', padx=(3,9))
            entry.bind('<Return>', lambda _: self.search())
        if observation:
            self.entries['observation'].insert(0, observation)
        ttk.Button(bar, text='Search', command=self.search).pack(side='left')
        left = ttk.Frame(panel)
        left.grid(row=1, column=0, sticky='nsew', padx=(0,8))
        self.listing = ttk.Treeview(left, show='headings', columns=('scene','cue','kind'), height=15)
        for name, width in (('scene',70), ('cue',58), ('kind',55)):
            self.listing.heading(name, text=name.capitalize())
            self.listing.column(name, width=width, stretch=False)
        self.listing.grid(row=0, column=0, sticky='nsew')
        left.rowconfigure(0, weight=1)
        scroll = ttk.Scrollbar(left, command=self.listing.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        self.listing.configure(yscrollcommand=scroll.set)
        self.listing.bind('<<TreeviewSelect>>', self.select)
        buttons = ttk.Frame(left)
        buttons.grid(row=1, column=0, columnspan=2)
        self.prev = ttk.Button(buttons, text='Previous', command=lambda: self.page(-1))
        self.next = ttk.Button(buttons, text='Next', command=lambda: self.page(1))
        self.prev.pack(side='left')
        self.next.pack(side='left')

    def text(self, value):
        self.notes.configure(state='normal')
        self.notes.delete('1.0', 'end')
        self.notes.insert('1.0', value)
        self.notes.configure(state='disabled')

    def clear_detail(self):
        self.refs, self.proofs = [], []
        self.preview_scope = None
        self.images.show('A', None, '')
        self.images.show('B', None, '')
        self.references.set('')
        self.references.configure(values=())
        self.proof_button.state(['disabled'])
        self.text('')

    def search(self):
        filters = {k: entry.get().strip() or None for k,entry in self.entries.items()}
        for key in ('scene','cue'):
            value = filters[key]
            if value is not None:
                if not value.isascii() or not value.isdecimal() or (key=='scene' and int(value)==0):
                    self.status.set('Enter a positive scene ID and a nonnegative cue index.')
                    return
                filters[key] = int(value) if key=='cue' else str(int(value))
        self.offset, self.filters = 0, filters
        self.page(0)

    def page(self, direction):
        self.offset = max(0, self.offset + direction * PAGE_SIZE)
        self.rows.clear()
        self.listing.delete(*self.listing.get_children())
        self.clear_detail()
        self.prev.state(['disabled'])
        self.next.state(['disabled'])
        self.status.set('Loading observations…')
        self.worker.request(partial(self.reader.page, offset=self.offset, **self.filters),
                            self.show_page, self.error)

    def show_page(self, result):
        rows, more = result
        for row in rows:
            item = self.listing.insert('', 'end', values=(row['scene'],row['cue'],row['kind']))
            self.rows[item] = row
        self.prev.state(['!disabled' if self.offset else 'disabled'])
        self.next.state(['!disabled' if more else 'disabled'])
        if rows:
            first = self.listing.get_children()[0]
            self.listing.selection_set(first)
            self.listing.focus(first)
        else:
            self.status.set('No observations match these filters.')

    def select(self, _event=None):
        selection = self.listing.selection()
        row = self.rows.get(selection[0]) if selection else None
        if row is None:
            return
        self.clear_detail()
        self.status.set('Loading ' + row['id'])
        self.worker.request(partial(self.load_detail, self.detail_reader, row['kind'], row['id']),
                            partial(self.show_detail, row), self.error)

    def load_detail(self, reader, kind, address):
        result = reader.detail(kind, address)
        if self.previews is not None:
            result['image'] = self.previews.image(kind, address, result['scope'], result['image'])
        return result

    def show_detail(self, row, result):
        self.preview_scope = result['scope']
        self.images.show('A', *result['image'])
        trace = result['trace']
        self.text(describe(trace) + describe_diagnostics(result))
        self.proofs.extend(result.get('recorded_decisions', []))
        for decision in trace['decisions']:
            self.refs.extend(decision['references'])
            self.proofs.extend(decision['merges'])
        self.refs.extend(result['associations'])
        labels = [f"{ref['label']} · {ref['kind']} · {ref['observation']}" for ref in self.refs]
        self.references.configure(values=labels)
        self.proof_button.state(['!disabled' if self.proofs else 'disabled'])
        self.status.set(f"Observation {row['id']} · {result['seconds']:g} s")
        if labels:
            self.references.current(0)
            self.show_reference()
        else:
            self.images.show('B', None, 'No recorded reference image')

    def show_reference(self, _event=None):
        index = self.references.current()
        if index < 0:
            return
        ref = self.refs[index]
        self.images.show('B', None, 'Loading reference…')
        self.worker.request(partial(self.load_reference, self.detail_reader, ref['kind'], ref['observation'], self.preview_scope),
                            lambda result: self.images.show('B', *result), self.error)

    def load_reference(self, reader, kind, address, scope):
        result = reader.reference(kind, address)
        return self.previews.image(kind, address, scope, result) if self.previews is not None else result

    def show_proofs(self):
        window = tk.Toplevel(self.root)
        window.title('Recorded evidence')
        notes = tk.Text(window, wrap='word', width=85, height=30)
        notes.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(window, command=notes.yview)
        scroll.pack(side='right', fill='y')
        notes.configure(yscrollcommand=scroll.set)
        notes.insert('1.0', json.dumps(self.proofs, indent=2, ensure_ascii=False))
        notes.configure(state='disabled')

    def error(self, error):
        self.status.set(str(error))

    def poll(self):
        self.worker.poll()
        self.poll_id = self.root.after(30, self.poll)

    def close(self):
        self.root.after_cancel(self.poll_id)
        self.worker.close(self.previews.close if self.previews is not None else lambda: None)
        self.panel.destroy()


def execute(args, config):
    scene_mode = args.command == 'inspect-scene'
    if scene_mode:
        from .scene_inspection import SceneReader
        from .scene_window import SceneInspector
        reader = SceneReader(config, args.scene_id, scope=args.scope)
    else:
        reader = IdentityReader(config, identity_id=args.identity_id, performer_id=args.performer_id, scope=args.scope)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError('The identity inspector requires a graphical desktop. Try python -m tkinter.') from error
    try:
        if scene_mode:
            from .inspection_crops import VttPreviews
            window = SceneInspector(root, reader, 'Scene ' + args.scene_id,
                                    previews=None if args.offline else VttPreviews(config))
        else:
            title = 'Performer ' + args.performer_id if args.performer_id else 'Identity ' + args.identity_id
            window = Inspector(root, reader, title, args.observation_id)
        root.protocol('WM_DELETE_WINDOW', root.quit)
        try:
            root.mainloop()
        finally:
            root.withdraw()
            window.close()
            del window
    finally:
        root.destroy()
        gc.collect()
    return 0
