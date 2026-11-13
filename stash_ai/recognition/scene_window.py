"""Lazy native tree reusing the identity inspector's image/evidence panel."""
from collections import deque
from functools import partial
from tkinter import ttk
import tkinter as tk

from .identity_window import Inspector
from .scene_inspection import SceneReader, UnassignedReader
from .inspection_runtime import Worker
from .inspection_navigation import History


class SceneInspector(Inspector):
    def build_browser(self, panel, observation):
        self.tree_worker = Worker('scene-inspector-tree')
        self.queue, self.branches, self.more_items = deque(), {}, {}
        self.groups = {}
        self.restart_items = {}
        self.view_filters={};self.history=History();self.restore_state=None;self.restoring=False
        bar = ttk.Frame(panel)
        bar.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0,8))
        ttk.Label(bar, text='Scene ID').pack(side='left')
        self.scene_entry = ttk.Entry(bar, width=12)
        self.scene_entry.insert(0, self.reader.scene)
        self.scene_entry.pack(side='left', padx=6)
        self.scene_entry.bind('<Return>', lambda _: self.search())
        ttk.Button(bar, text='Load / refresh', command=self.search).pack(side='left')
        ttk.Button(bar,text='Search / filters',command=self.filter_dialog).pack(side='left')
        ttk.Button(bar,text='Back',command=lambda:self.navigate(-1)).pack(side='left')
        ttk.Button(bar,text='Forward',command=lambda:self.navigate(1)).pack(side='left')
        self.summary = ttk.Label(bar)
        self.summary.pack(side='left', padx=10)
        left = ttk.Frame(panel)
        left.grid(row=1, column=0, sticky='nsew', padx=(0,8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.listing = ttk.Treeview(left, show='tree headings', columns=('scene','cue','kind','state'), height=18)
        self.listing.heading('#0', text='Performer / observation')
        self.listing.column('#0', width=240, minwidth=120)
        for name, width in (('scene',55), ('cue',40), ('kind',45), ('state',100)):
            self.listing.heading(name, text=name.capitalize())
            self.listing.column(name, width=width, minwidth=35, stretch=False)
        self.listing.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(left, command=self.listing.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        self.listing.configure(yscrollcommand=scroll.set)
        self.listing.bind('<<TreeviewOpen>>', self.expand)
        self.listing.bind('<<TreeviewSelect>>', self.select)
        self.listing.tag_configure('entry_scene', foreground='#2466a0')
        ttk.Label(left, text='Expand a group. Blue rows belong to the entry scene.\nSelect “Load more…” to read the next page.').grid(row=1, column=0, sticky='w')

    def search(self):
        scene = self.scene_entry.get().strip()
        if not scene.isascii() or not scene.isdecimal() or int(scene)<=0:
            self.status.set('Enter a positive scene ID.')
            return
        if not self.restoring and self.rows:
            self.restore_state=self.capture()
            self.history.record(self.restore_state)
            if scene!=self.reader.scene:self.restore_state=None
        self.reader = SceneReader(self.reader.config, str(int(scene)), scope=self.reader.scope)
        self.worker.invalidate()
        self.tree_worker.invalidate()
        self.queue.clear()
        self.branches.clear()
        self.groups.clear()
        self.more_items.clear()
        self.restart_items.clear()
        self.rows.clear()
        self.listing.delete(*self.listing.get_children())
        self.clear_detail()
        self.root.title('Scene identity inspector · Scene ' + self.reader.scene)
        self.status.set('Loading scene identities…')
        self.summary.configure(text='')
        self.tree_worker.request(self.reader.catalogue, self.show_catalogue, self.error)

    def show_catalogue(self, result):
        self.scan = result['scan']
        self.summary.configure(text=f"Scope {self.scan['scope']} · {self.scan['status']} scan")
        if self.view_filters:
            from .inspection_search import SearchReader
            reader=SearchReader(self.reader,self.view_filters,scan=self.scan)
            self.add_branch('Search results',reader,True,'Filtered across the complete view',level='search')
            item=next(reversed(self.branches))
            self.listing.item(item,open=True);self.load_branch(item)
            return
        for group in result['groups']:
            label = (f"Performer {group['performer']}" if group['performer'] else
                     f"Identity {group['identity'][:12]} · no Stash performer")
            item = self.listing.insert('', 'end', text=label + f" · {group['count']} in scene")
            description = f"Identity {group['identity']}\nAll retained scenes. Expand Face or Body to load observations."
            self.groups[item] = description
            reader = self.reader.identity_reader(group['identity'], self.scan['scope'])
            for kind in ('face', 'body'):
                self.add_branch(kind.capitalize(), reader, False,
                                f"{kind.capitalize()} observations\n{description}", parent=item, kind=kind, level='states')
        reader = UnassignedReader(self.reader.config, self.scan)
        item = self.listing.insert('', 'end', text=f"Unassigned in this scene · {result['unassigned']}")
        description = ('No accepted face or body membership. Includes provisional body tracks.\n'
                       'Only retained observations from this scene scan are available.')
        self.groups[item] = description
        for kind in ('face', 'body'):
            kind_item = self.listing.insert(item, 'end', text=kind.capitalize())
            self.groups[kind_item] = f'{kind.capitalize()} observations\n{description}'
            states = [g for g in result['unassigned_groups'] if g['kind']==kind]
            for group in states:
                self.add_branch(f"{group['state']} · {group['count']}", reader, True,
                                f"{kind.capitalize()} observations · State: {group['state']}\n{description}",
                                parent=kind_item, kind=kind, state=group['state'])
            if not states:
                self.listing.insert(kind_item, 'end', text='No observations in this group.')
        self.status.set('Expand a performer to see all scenes, or inspect unassigned observations from this scene.')
        self.restore_view()

    def add_branch(self, label, reader, unassigned, description, *, parent='', kind=None, state=None, level='observations'):
        item = self.listing.insert(parent, 'end', text=label)
        marker = self.listing.insert(item, 'end', text='Expand to load…')
        self.branches[item] = dict(reader=reader, unassigned=unassigned, kind=kind, state=state, level=level, offset=0,
                                   loading=False, loaded=False, marker=marker, description=description, after=None)

    def expand(self, _event=None):
        item = self.listing.focus()
        if item in self.branches and not self.branches[item]['loaded']:
            self.load_branch(item)

    def load_branch(self, item):
        branch = self.branches[item]
        if branch['loading']:
            return
        branch['loading'] = True
        self.listing.item(branch['marker'], text='Loading…')
        self.queue.append(item)

    def show_branch(self, item, result):
        branch = self.branches[item]
        if branch['level'] == 'states':
            self.more_items.pop(branch['marker'], None)
            self.listing.delete(branch['marker'])
            for group in result:
                self.add_branch(f"{group['state']} · {group['count']}", branch['reader'], False,
                    f"{branch['kind'].capitalize()} observations · State: {group['state']}\n"
                    'Recorded membership role across all retained scenes; not current quality eligibility.',
                    parent=item, kind=branch['kind'], state=group['state'])
            if not result:
                self.listing.insert(item, 'end', text='No observations in this group.')
            branch.update(loaded=True, loading=False)
            self.status.set('Expand a state to load observations across all scenes.')
            self.restore_view()
            return
        rows, more = result
        marker = branch['marker']
        self.more_items.pop(marker, None)
        self.listing.delete(marker)
        for row in rows:
            state = row.get('state') or (row['role'] if row['status']=='accepted' else row['status'])
            child = self.listing.insert(item, 'end', text=row['id'],
                values=(row['scene'],row['cue'],row['kind'],state),
                tags=('entry_scene',) if row['scene']==self.reader.scene else ())
            self.rows[child] = dict(row, reader=branch['reader'].owner(row) if branch['level']=='search' else branch['reader'])
        branch.update(offset=branch['offset']+len(rows), loaded=True, loading=False)
        if rows: branch['after']=rows[-1]
        self.trim_rows()
        if more:
            marker = branch['marker'] = self.listing.insert(item, 'end', text='Load more…')
            self.more_items[marker] = item
        elif not rows and not branch['offset']:
            self.listing.insert(item, 'end', text='No observations in this group.')
        self.status.set(f"Loaded {branch['offset']} observations in this group" + (' · more available' if more else ''))
        self.restore_view()

    def trim_rows(self):
        while len(self.rows)>1000:
            child=next(iter(self.rows))
            parent=self.listing.parent(child)
            self.rows.pop(child)
            self.listing.delete(child)
            if parent in self.branches and parent not in self.restart_items.values():
                marker=self.listing.insert(parent,0,text='Earlier observations…')
                self.restart_items[marker]=parent

    def branch_error(self, item, error):
        branch = self.branches[item]
        branch['loading'] = False
        self.listing.item(branch['marker'], text='Load failed · select to retry')
        self.more_items[branch['marker']] = item
        self.error(error)

    def select(self, _event=None):
        selected = self.listing.selection()
        item = selected[0] if selected else None
        self.worker.invalidate()
        self.clear_detail()
        if item in self.restart_items:
            parent=self.restart_items.pop(item)
            for child in self.listing.get_children(parent):
                self.rows.pop(child,None)
                self.more_items.pop(child,None)
                self.listing.delete(child)
            branch=self.branches[parent]
            branch.update(offset=0,after=None,marker=self.listing.insert(parent,'end',text='Loading…'))
            self.load_branch(parent)
        elif item in self.more_items:
            self.load_branch(self.more_items[item])
        elif item in self.groups:
            self.text(self.groups[item])
        elif item in self.branches:
            self.text(self.branches[item]['description'])
        elif item in self.rows:
            self.detail_reader = self.rows[item]['reader']
            super().select()
            if not self.restoring:self.history.record(self.capture())

    def poll(self):
        self.tree_worker.poll()
        if self.queue and self.tree_worker.future is None:
            item = self.queue.popleft()
            branch = self.branches[item]
            if branch['level'] == 'states':
                task = partial(branch['reader'].states, branch['kind'])
            else:
                args = (dict(offset=branch['offset']) if branch['level']=='search' else
                        dict(after=branch['after'], kind=branch['kind'], state=branch['state']))
                if not branch['unassigned']:
                    args['accepted_only'] = True
                task = partial(branch['reader'].page, **args)
            self.tree_worker.request(task,
                                     partial(self.show_branch, item), partial(self.branch_error, item))
        super().poll()

    def filter_dialog(self):
        from .inspection_search import SORTS
        window=tk.Toplevel(self.root);window.title('Search this view')
        entries={}
        choices={'kind':['All','face','body'],'state':['All','anchor','auxiliary','attachment','unknown','unassigned','provisional','withheld'],
                 'policy':['All','current','historical','missing'],'assignment':['All','assigned','unassigned'],'sort':list(SORTS)}
        for i,(key,label) in enumerate((('text','Observation ID contains'),('performer','Performer ID'),('scene','Scene ID'),('kind','Kind'),('state','State / rejection'),('reason','Reason contains'),('policy','Policy'),('max_margin','Maximum recorded lead'),('sort','Sort'),('assignment','Membership'))):
            ttk.Label(window,text=label).grid(row=i,column=0,sticky='w',padx=6,pady=3)
            widget=ttk.Combobox(window,values=choices[key]) if key in choices else ttk.Entry(window)
            widget.insert(0,str(self.view_filters.get(key,'')))
            widget.grid(row=i,column=1,padx=6,pady=3);entries[key]=widget
        missing=tk.BooleanVar(master=window,value=self.view_filters.get('missing_crop',False))
        ttk.Checkbutton(window,text='Missing retained crop',variable=missing).grid(row=10,column=0,columnspan=2)
        def apply(clear=False):
            self.view_filters={} if clear else {k:w.get().strip() for k,w in entries.items() if w.get().strip() and w.get().strip()!='All'}
            if not clear and missing.get():self.view_filters['missing_crop']=True
            self.restore_state=None;self.restoring=True;self.search();self.restoring=False;window.destroy()
        ttk.Button(window,text='Apply',command=apply).grid(row=11,column=0)
        ttk.Button(window,text='Clear filters',command=lambda:apply(True)).grid(row=11,column=1)

    def branch_key(self,branch):
        reader=branch['reader']
        return (getattr(reader,'selector',('unassigned',))[1] if hasattr(reader,'selector') and not branch['unassigned'] else 'unassigned',branch['kind'],branch['state'])

    def capture(self):
        row=self.current_row
        return dict(scene=self.reader.scene,filters=dict(self.view_filters),selected=(row['kind'],row['id']) if row else None,
                    row=dict(row) if row else None,open=[self.branch_key(b) for k,b in self.branches.items() if self.listing.item(k,'open')],
                    scroll=self.listing.yview()[0],images=self.images.view_state())

    def navigate(self,direction):
        if self.history.index>=0:self.history.entries[self.history.index]=self.capture()
        state=self.history.move(direction)
        if state:
            state=dict(state);state.pop('selection_requested',None);state.pop('restored_panels',None)
            self.restoring=True;self.restore_state=state;self.view_filters=dict(state['filters'])
            self.scene_entry.delete(0,'end');self.scene_entry.insert(0,state['scene']);self.search()

    def restore_view(self):
        state=self.restore_state
        if not state:return
        for item,branch in list(self.branches.items()):
            if self.branch_key(branch) in state['open']:
                self.listing.item(item,open=True)
                parent=self.listing.parent(item)
                if parent:self.listing.item(parent,open=True)
                if not branch['loaded'] and not branch['loading']:self.load_branch(item)
        self.listing.yview_moveto(state['scroll'])
        selected=state.get('selected')
        if selected and not state.get('selection_requested'):
            state['selection_requested']=True
            row=state['row'];reader=row['reader']
            self.worker.request(partial(reader.page,kind=row['kind'],observation=row['id']),
                                partial(self.restore_selection,state),self.error)
        elif not selected:self.restoring=False

    def restore_selection(self,state,result):
        rows,_=result
        if not rows:
            self.status.set('Previous selection is no longer a member. Refresh completed.');self.restoring=False;return
        row=dict(rows[0],reader=state['row']['reader'])
        child=next((k for k,v in self.rows.items() if (v['kind'],v['id'])==state['selected']),None)
        if child is None:
            parent=self.listing.insert('','end',text='Restored selection',open=True)
            child=self.listing.insert(parent,'end',text=row['id'],values=(row['scene'],row['cue'],row['kind'],row['state']))
            self.rows[child]=row
        self.listing.selection_set(child);self.listing.see(child)
        self.root.after_idle(lambda:setattr(self,'restoring',False))

    def display_preview(self,panel,result):
        super().display_preview(panel,result)
        if (self.restore_state and self.restore_state.get('images') and self.current_row and
                (self.current_row['kind'],self.current_row['id'])==self.restore_state.get('selected') and
                panel not in self.restore_state.get('restored_panels',set())):
            self.images.restore(self.restore_state['images'])
            self.restore_state.setdefault('restored_panels',set()).add(panel)

    def close(self):
        self.tree_worker.close(lambda: None)
        super().close()
