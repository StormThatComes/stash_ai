"""Native batch comparison, scoped preflight and replay-backed YAML refinement."""
from collections import deque
from functools import partial
import time
import tkinter as tk
from tkinter import ttk

from .inspection_pair_replay import experiment
from .inspection_pair_variants import preflight
from .inspection_runtime import Worker


class VariantsWindow:
    def __init__(self, comparison):
        self.comparison = comparison
        self.config, self.scan = comparison.reader.config, comparison.scan
        self.ids = comparison.pair() + ((comparison.protected,) if comparison.protected else ())
        self.revision = comparison.revision
        self.worker = Worker('inspect-pair-variants')
        self.live = deque(maxlen=1)
        self.window = tk.Toplevel(comparison.window)
        self.window.title('Compare YAML variants')
        self.window.geometry('1040x610')
        self.full = tk.BooleanVar(master=self.window)
        bar = ttk.Frame(self.window,padding=5)
        bar.pack(fill='x')
        for label, value in (('Entry and identity scenes',False),('Full scope',True)):
            ttk.Radiobutton(bar,text=label,variable=self.full,value=value,command=self.prepare).pack(side='left')
        self.estimate = ttk.Label(self.window,padding=4)
        self.estimate.pack(anchor='w')
        self.options = {}
        self.choices = tk.Listbox(self.window,selectmode='extended',exportselection=False,height=7)
        self.choices.pack(fill='x',padx=5)
        self.choices.bind('<<ListboxSelect>>', self.update_actions)
        selected = comparison.tree.selection()
        for key,(route,patch) in comparison.items.items():
            if patch is not None:
                n = self.add(route['title'] + ' · ' + patch['title'],patch)
                if key in selected:
                    self.choices.selection_set(n)
        if self.options and not self.choices.curselection():
            self.choices.selection_set(0)
        controls = ttk.Frame(self.window,padding=4)
        controls.pack(fill='x')
        self.run = ttk.Button(controls,text='Compare selected variants',command=self.compare)
        self.run.pack(side='left')
        self.cancel = ttk.Button(controls,text='Cancel',command=self.prepare)
        self.cancel.pack(side='left')
        self.status = ttk.Label(controls)
        self.status.pack(side='left',padx=4)
        cols = ('result','join','protected','withheld','extra','yaml','seconds')
        self.table = ttk.Treeview(self.window,columns=cols,show='headings',selectmode='browse')
        for col,label,width in zip(cols,('Variant','A+B fully joined','C separate','Unassigned¹','Extra identities','YAML keys','Seconds'),
                                   (145,120,90,145,110,90,85)):
            self.table.heading(col,text=label)
            self.table.column(col,width=width,minwidth=50)
        self.table.pack(fill='both',expand=True,padx=5)
        self.table.bind('<<TreeviewSelect>>',self.update_actions)
        self.table.bind('<Double-1>',self.details)
        self.footer = ttk.Frame(self.window,padding=5)
        self.footer.pack(fill='x')
        self.diff = ttk.Button(self.footer,text='Observation changes…',command=self.details)
        self.copy = ttk.Button(self.footer,text='Copy tested YAML',command=self.copy_yaml)
        self.validate = ttk.Button(self.footer,text='Validate on full scope',command=self.validate_full)
        self.minimize = ttk.Button(self.footer,text='Minimize selected patch',command=self.refine)
        for button in (self.diff,self.copy,self.validate,self.minimize):
            button.pack(side='left')
        ttk.Label(self.footer,text='Probe budget').pack(side='left',padx=4)
        self.budget = tk.StringVar(master=self.window,value='16')
        ttk.Entry(self.footer,textvariable=self.budget,width=4).pack(side='left')
        ttk.Label(self.window,text='¹ All face and body observations in the selected replay scenes. Extra identities share a destination with A or B.').pack(anchor='w')
        self.plan = self.result = None
        self.patches = {}
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.prepare()

    def add(self, label, patch):
        for i, (old, _) in enumerate(self.options.values()):
            if old['yaml'] == patch['yaml']:
                return i
        index = len(self.options)
        name = f'V{index+1}'
        self.options[name] = (patch,label)
        self.choices.insert('end', name + ' · ' + label)
        return index

    def prepare(self, then=None):
        self.worker.invalidate()
        self.plan = self.result = None
        self.live = deque(maxlen=1)
        self.busy = True
        self.table.delete(*self.table.get_children())
        self.estimate.configure(text='Counting retained replay inputs…')
        self.status.configure(text='')
        self.update_actions()
        def ready(plan):
            self.plan, self.busy = plan, False
            self.estimate.configure(text=f"{self.scope_label()} · {len(plan['scenes'])} scenes · {plan['faces']} face and {plan['bodies']} body observations")
            self.update_actions()
            if then:
                then()
        self.worker.request(partial(preflight,self.config,self.scan['scope'],self.ids,self.scan['scene'],self.full.get(),self.revision),
                            ready,self.failed)

    def scope_label(self):
        return 'Full scope' if self.full.get() else 'Quick replay'

    def update_actions(self, event=None):
        ready = self.plan is not None and not self.busy
        selection = self.table.selection()
        name = selection[0] if selection else None
        patch = self.patches.get(name)
        verified = self.result and name in self.result['trials'] and self.result['trials'][name]['success']
        self.run.state(['!disabled'] if ready and self.choices.curselection() else ['disabled'])
        self.cancel.state(['!disabled'] if self.busy else ['disabled'])
        self.diff.state(['!disabled'] if self.result and name else ['disabled'])
        self.copy.state(['!disabled'] if ready and patch and patch['yaml'] else ['disabled'])
        self.minimize.state(['!disabled'] if ready and verified and patch and patch['yaml'] else ['disabled'])
        self.validate.state(['!disabled'] if ready and patch is not None and not self.full.get() else ['disabled'])

    def compare(self):
        selected = set(self.choices.curselection())
        self.execute({name:patch for i,(name,(patch,_)) in enumerate(self.options.items()) if i in selected})

    def execute(self, patches, budget=None):
        self.busy = True
        self.result = None
        self.patches = dict(patches)
        self.table.delete(*self.table.get_children())
        self.update_actions()
        live = self.live = deque(maxlen=1)
        start = time.monotonic()
        self.status.configure(text='Preparing immutable replay inputs…')
        def progress(name, stage, item):
            live.append(f'{name} · {stage} · {time.monotonic()-start:.0f}s')
        def done(result):
            self.busy = False
            self.result = result
            if 'minimized' in result:
                self.patches['minimized'] = result['minimized']
                self.add('Minimized YAML',result['minimized'])
            for name,trial in result['trials'].items():
                patch = self.patches.get(name, {})
                c = trial['protected']
                self.table.insert('', 'end', iid=name,values=(name,'Yes' if trial['joined'] else 'No',
                    '—' if c is None else 'Yes' if c else 'No',trial['unassigned'],len(trial['outsiders']),
                    len(patch.get('changes',())),f"{trial['elapsed']:.2f}"))
            best = 'minimized' if 'minimized' in result else next(iter(patches))
            self.table.selection_set(best)
            message = f'{self.scope_label()} · complete'
            if 'minimized' in result:
                message += f" · {result['probe_count']} probes · " + ('probe budget reached' if result['budget_reached'] else 'local search finished')
            self.status.configure(text=message)
            self.update_actions()
        self.worker.request(partial(experiment,self.config,self.scan['scope'],self.ids,patches,self.revision,
                                    scenes=self.plan['scenes'],progress=progress,refine_budget=budget),done,self.failed)

    def poll(self):
        self.worker.poll()
        if self.busy and self.live:
            self.status.configure(text=self.live.pop())

    def failed(self, error):
        self.busy = False
        self.status.configure(text=str(error),wraplength=650)
        if self.plan is None:
            self.estimate.configure(text='Preflight failed')
        self.update_actions()

    def details(self, event=None):
        selected = self.table.selection()
        if self.result and selected:
            self.comparison.show_changes(self.result,selected[0])

    def copy_yaml(self):
        text = self.patches[self.table.selection()[0]]['yaml']
        self.window.clipboard_clear()
        self.window.clipboard_append(text)

    def refine(self):
        try:
            budget = int(self.budget.get())
            if budget < 1:
                raise ValueError
        except ValueError:
            self.status.configure(text='Enter a positive integer for the probe budget.')
            return
        self.execute({'option':self.patches[self.table.selection()[0]]},budget)

    def validate_full(self):
        patch = self.patches[self.table.selection()[0]]
        self.full.set(True)
        self.prepare(lambda:self.execute({'option':patch}))

    def close(self):
        self.worker.close()
        self.window.destroy()
        self.comparison.variants_window = None
