"""Paged appearance comparisons using retained images and production measurements."""
from contextlib import closing
from functools import partial
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

from .inspection_runtime import Worker, open_readonly, check_cancelled, stamp
from .inspection_pair_evidence import face_witnesses
from .inspection_view_evidence import face_rows, formation, projection_image


class ViewsWindow:
    size = 4

    def __init__(self, comparison):
        self.parent, self.host, self.reader = comparison, comparison.host, comparison.reader
        self.scope, self.ids = comparison.scan['scope'], comparison.pair()
        self.window = tk.Toplevel(comparison.window)
        self.window.title('View and Hybrid evidence')
        self.window.geometry('1050x660')
        self.worker = Worker('appearance-inspector',resource=self.host.previews.fork if self.host.previews else None)
        self.tabs = ttk.Notebook(self.window)
        self.tabs.pack(fill='both',expand=True,padx=6,pady=6)
        self.pages = [ttk.Frame(self.tabs,padding=6) for _ in range(3)]
        for frame, label in zip(self.pages,('View matrix','View formation','Hybrid witnesses')):
            self.tabs.add(frame,text=label)
        self.start = [0,0]
        self.photos = []
        self.status = ttk.Label(self.window,wraplength=1000)
        self.status.pack(fill='x')
        self.grid = ttk.Frame(self.pages[0])
        self.grid.pack()
        bar=ttk.Frame(self.pages[0])
        bar.pack()
        self.nav = []
        for axis,letter in enumerate('AB'):
            for sign in (-1,1):
                button=ttk.Button(bar,text=('Previous ' if sign<0 else 'Next ')+letter,
                                  command=partial(self.move,axis,sign))
                button.pack(side='left')
                self.nav.append(button)
        self.weak = ttk.Button(bar,text='Inspect weakest pair',command=self.open_pair,state='disabled')
        self.weak.pack(side='left')
        self.rival = ttk.Button(bar,text='Inspect component rival',command=self.open_rival,state='disabled')
        self.rival.pack(side='left')
        self.build_formation()
        self.tabs.bind('<<NotebookTabChanged>>',self.refresh)
        self.window.protocol('WM_DELETE_WINDOW',self.close)

    def request(self, work, done):
        def read():
            if stamp(self.reader.config.database)!=self.parent.revision:
                raise ValueError('Recognition changed. Refresh the comparison.')
            return work()
        self.worker.request(read,done,lambda e:self.status.configure(text=str(e)))

    def refresh(self,event=None):
        self.worker.invalidate()
        self.photos.clear()
        page=self.tabs.index('current')
        (self.matrix,self.samples,self.hybrid)[page]()

    def thumbnail(self, kind, addr, size):
        if not addr:
            return None,'No witness'
        if kind=='person':
            im,caption=projection_image(self.reader.config,self.scope,addr)
        else:
            from .inspection_context import locate
            row=locate(self.reader.config,kind,addr,self.scope)
            data=self.host.load_reference(row['reader'],kind,addr,self.scope)
            im,caption=data['annotated']
        if im is not None:
            im=im.copy()
            im.thumbnail(size)
        return im,caption

    def draw_image(self, label, image):
        if image is not None:
            photo=ImageTk.PhotoImage(image,master=self.window)
            self.photos.append(photo)
            label.configure(image=photo,compound='top')

    def move(self, axis, sign):
        self.start[axis] += sign*self.size
        self.matrix()

    def matrix(self):
        for w in self.grid.winfo_children(): w.destroy()
        self.photos.clear()
        matrix=self.parent.matrix
        axes=matrix.get('axes',[[],[]])
        selected=[v[n:n+self.size] for v,n in zip(axes,self.start)]
        cells={tuple(p['medoids']):p for p in matrix.get('pairs',())}
        compared=set(map(tuple,matrix.get('tested',())))
        self.status.configure(text=f'{len(cells)} measured pairs · {len(compared)} pairs probed for YAML changes '
                              '· choose a cell to inspect its score, blocker and witnesses.')
        self.weak.state(['disabled'])
        component=matrix.get('component') or {}
        rival=component.get('nearest_rival') or {}
        self.rival.state(['!disabled'] if rival.get('medoids') else ['disabled'])
        for axis in range(2):
            self.nav[2*axis].state(['!disabled'] if self.start[axis]>0 else ['disabled'])
            self.nav[2*axis+1].state(['!disabled'] if self.start[axis]+self.size<len(axes[axis]) else ['disabled'])
        labels=[]
        for axis,views in enumerate(selected):
            for index,v in enumerate(views,1):
                label=ttk.Label(self.grid,text=f"{'AB'[axis]}{self.start[axis]+index} · {v['pose']}\n{v['support']} votes")
                label.grid(row=index if axis==0 else 0,column=0 if axis==0 else index,padx=4,pady=4)
                labels.append((label,v['id']))
        for i,a in enumerate(selected[0],1):
            for j,b in enumerate(selected[1],1):
                key=(a['id'],b['id'])
                p=cells[key]
                text=f"{p['match_score']:.3f} / {p['threshold']:.3f}\n"+(p['blockers'][0] if p['blockers'] else 'Pair gates pass')
                button=tk.Button(self.grid,text=text,wraplength=125,background='#f5d7c8' if p['blockers'] else '#d7eadc',
                                 command=partial(self.cell,p,key in compared))
                button.grid(row=i,column=j,sticky='nsew')
        def thumbnails():
            return [self.thumbnail('face',addr,(64,50))[0] for _,addr in labels]
        self.request(thumbnails,lambda imgs:[self.draw_image(label,img) for (label,_),img in zip(labels,imgs)])

    def cell(self, proof, tested):
        self.selected=proof
        self.weak.state(['!disabled'])
        self.status.configure(text=f"Score {proof['match_score']:.4f} / {proof['threshold']:.4f} · lowest {proof['minimum_cross_similarity']:.4f} "
            f"· independent votes {proof['support']} · YAML probed: {'yes' if tested else 'no'} · "
            + (', '.join(proof['blockers']) or 'Pair gates pass. Component and rivals are checked separately.'))
        self.open_pair()

    def open_pair(self):
        proof=self.selected
        def evidence():
            with closing(open_readonly(self.reader.config.database)) as db:
                refs=face_witnesses(db,self.reader.config,self.ids,proof,None)
            return next((w for w in refs if w['label']=='Weakest cross comparison'),refs[0])
        self.request(evidence,self.parent.view)

    def open_rival(self):
        refs=self.parent.matrix['component']['nearest_rival']['medoids']
        self.parent.view(dict(label='Closest component rival',refs=[('face',k) for k in refs]))

    def build_formation(self):
        page=self.pages[1]
        bar=ttk.Frame(page)
        bar.pack(fill='x')
        self.owner=ttk.Combobox(bar,values=['Identity A','Identity B'],state='readonly',width=14)
        self.owner.current(0)
        self.owner.pack(side='left')
        self.owner.bind('<<ComboboxSelected>>',self.change_owner)
        self.offset=0
        self.picked=set()
        self.rows=[]
        self.prev=ttk.Button(bar,text='Previous',command=lambda:self.turn(-1))
        self.next=ttk.Button(bar,text='Next',command=lambda:self.turn(1))
        for b in (self.prev,self.next): b.pack(side='left')
        ttk.Button(bar,text='Explain selected faces',command=self.explain).pack(side='left')
        self.list=tk.Listbox(page,selectmode='extended',exportselection=False,height=12)
        self.list.pack(fill='x')
        self.notes=tk.Text(page,wrap='word',height=12)
        self.notes.pack(fill='both',expand=True)

    def change_owner(self,event=None):
        self.offset=0
        self.picked.clear()
        self.samples()

    def remember(self):
        self.picked.difference_update(r['id'] for r in self.rows)
        self.picked.update(self.rows[i]['id'] for i in self.list.curselection())

    def turn(self, sign):
        self.remember()
        self.offset+=80*sign
        self.samples()

    def samples(self):
        self.rows=[]
        self.list.delete(0,'end')
        self.prev.state(['disabled'])
        self.next.state(['disabled'])
        self.status.configure(text='Select faces from one identity with Ctrl or Shift. Selections are kept between pages.')
        def display(rows):
            self.rows=rows[:80]
            for i,row in enumerate(self.rows):
                self.list.insert('end',f"Scene {row['scene']} · cue {row['cue']} · {row['id']}")
                if row['id'] in self.picked:
                    self.list.selection_set(i)
            self.prev.state(['!disabled'] if self.offset else ['disabled'])
            self.next.state(['!disabled'] if len(rows)>80 else ['disabled'])
        self.request(partial(face_rows,self.reader,self.scope,self.ids[self.owner.current()],offset=self.offset,limit=81),display)

    def explain(self):
        self.remember()
        if not self.picked:
            self.status.configure(text='Select the faces to evaluate together.')
            return
        def show(msg):
            self.notes.delete('1.0','end')
            self.notes.insert('end',msg)
        self.request(partial(formation,self.reader,self.scope,self.ids[self.owner.current()],tuple(self.picked)),show)

    def hybrid(self):
        page=self.pages[2]
        for w in page.winfo_children(): w.destroy()
        hybrid=self.parent.hybrid
        if not hybrid:
            self.status.configure(text='No comparable hybrid evidence for this pair under the current configuration.')
            return
        tasks=[]
        for i,p in enumerate(hybrid['panels']):
            value='unknown' if p['minimum'] is None else f"{p['minimum']:.4f}"
            for j,addr in enumerate(p['ids']):
                label=ttk.Label(page,text=f"{'AB'[j]} · {p['signal']} · {p['label']}")
                label.grid(row=2*i,column=j,padx=5,pady=3)
                tasks.append((label,p['signal'],addr))
            ttk.Label(page,text=f"Minimum {value} / required {p['floor']:.4f}").grid(row=2*i+1,column=0,columnspan=2)
        rival=hybrid['rival'] or {}
        self.status.configure(text=f"Rival {rival.get('rival','unavailable')} · lead {rival.get('gap','unavailable')} "
                              f"/ required {hybrid['margin']:g} · "+(', '.join(hybrid['blockers']) or 'Pair gates pass'))
        def display(results):
            for (label,_,_),(im,caption) in zip(tasks,results):
                label.configure(text=label.cget('text')+'\n'+caption)
                self.draw_image(label,im)
        self.request(lambda:[self.thumbnail(k,addr,(360,190)) for _,k,addr in tasks],display)

    def close(self):
        self.worker.close()
        self.window.destroy()
        self.parent.views_window=None
