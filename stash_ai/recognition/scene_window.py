"""Lazy native tree reusing the identity inspector's image/evidence panel."""
from collections import deque
from functools import partial
from tkinter import ttk

from .identity_window import Inspector
from .scene_inspection import SceneReader, UnassignedReader
from .tune_worker import Worker


class SceneInspector(Inspector):
    def build_browser(self, panel, observation):
        self.tree_worker = Worker('scene-inspector-tree')
        self.queue, self.branches, self.more_items = deque(), {}, {}
        self.groups = {}
        bar = ttk.Frame(panel)
        bar.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0,8))
        ttk.Label(bar, text='Scene ID').pack(side='left')
        self.scene_entry = ttk.Entry(bar, width=12)
        self.scene_entry.insert(0, self.reader.scene)
        self.scene_entry.pack(side='left', padx=6)
        self.scene_entry.bind('<Return>', lambda _: self.search())
        ttk.Button(bar, text='Load / refresh', command=self.search).pack(side='left')
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
        self.reader = SceneReader(self.reader.config, str(int(scene)), scope=self.reader.scope)
        self.worker.invalidate()
        self.tree_worker.invalidate()
        self.queue.clear()
        self.branches.clear()
        self.groups.clear()
        self.more_items.clear()
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
        for group in result['groups']:
            label = (f"Performer {group['performer']}" if group['performer'] else
                     f"Identity {group['identity'][:12]} · no Stash performer")
            item = self.listing.insert('', 'end', text=label + f" · {group['count']} in scene")
            description = f"Identity {group['identity']}\nAll retained scenes. Expand Face or Body to load observations."
            self.groups[item] = description
            reader = self.reader.identity_reader(group['identity'], self.scan['scope'])
            for kind in ('face', 'body'):
                self.add_branch(kind.capitalize(), reader, False,
                                f"{kind.capitalize()} observations\n{description}", parent=item, kind=kind)
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

    def add_branch(self, label, reader, unassigned, description, *, parent='', kind=None, state=None):
        item = self.listing.insert(parent, 'end', text=label)
        marker = self.listing.insert(item, 'end', text='Expand to load…')
        self.branches[item] = dict(reader=reader, unassigned=unassigned, kind=kind, state=state, offset=0,
                                   loading=False, loaded=False, marker=marker, description=description)

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
        rows, more = result
        marker = branch['marker']
        self.more_items.pop(marker, None)
        self.listing.delete(marker)
        for row in rows:
            state = row.get('state') or (row['role'] if row['status']=='accepted' else row['status'])
            child = self.listing.insert(item, 'end', text=row['id'],
                values=(row['scene'],row['cue'],row['kind'],state),
                tags=('entry_scene',) if row['scene']==self.reader.scene else ())
            self.rows[child] = dict(row, reader=branch['reader'])
        branch.update(offset=branch['offset']+len(rows), loaded=True, loading=False)
        if more:
            marker = branch['marker'] = self.listing.insert(item, 'end', text='Load more…')
            self.more_items[marker] = item
        elif not rows and not branch['offset']:
            self.listing.insert(item, 'end', text='No observations in this group.')
        self.status.set(f"Loaded {branch['offset']} observations in this group" + (' · more available' if more else ''))

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
        if item in self.more_items:
            self.load_branch(self.more_items[item])
        elif item in self.groups:
            self.text(self.groups[item])
        elif item in self.branches:
            self.text(self.branches[item]['description'])
        elif item in self.rows:
            self.detail_reader = self.rows[item]['reader']
            super().select()

    def poll(self):
        self.tree_worker.poll()
        if self.queue and self.tree_worker.future is None:
            item = self.queue.popleft()
            branch = self.branches[item]
            args = dict(offset=branch['offset'], kind=branch['kind'])
            if branch['unassigned']:
                args['state'] = branch['state']
            else:
                args['accepted_only'] = True
            self.tree_worker.request(partial(branch['reader'].page, **args),
                                     partial(self.show_branch, item), partial(self.branch_error, item))
        super().poll()

    def close(self):
        self.tree_worker.close(lambda: None)
        super().close()
