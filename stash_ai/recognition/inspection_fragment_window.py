"""Inspect both directions and the entry scene's body-track evidence by cue."""
import tkinter as tk
from tkinter import ttk
from functools import partial

from .inspection_attachment import simulate, timeline
from .inspection_runtime import Worker, stamp


class FragmentWindow:
    def __init__(self, parent):
        self.parent=parent
        self.worker=Worker('pair-fragment-scenarios')
        self.window=tk.Toplevel(parent.window)
        self.window.title('Attachment scenarios and track timeline')
        self.window.geometry('1000x610')
        bar=ttk.Frame(self.window)
        bar.pack(fill='x')
        self.direction=ttk.Combobox(bar,values=('A → B','B → A'),state='readonly',width=8)
        self.direction.current(1)
        self.direction.pack(side='left')
        self.family=ttk.Combobox(bar,values=('Confirmed View','Hybrid query','Body-track timeline'),state='readonly',width=20)
        self.family.current(0)
        self.family.pack(side='left')
        self.run=ttk.Button(bar,text='Run selected scenario',command=self.load)
        self.run.pack(side='left')
        ttk.Button(bar,text='Cancel',command=self.clear).pack(side='left')
        self.only=tk.BooleanVar(master=self.window,value=False)
        ttk.Checkbutton(bar,text='Only exceptions / track blockers',variable=self.only,command=self.filter).pack(side='left')
        self.status=ttk.Label(self.window,wraplength=950,padding=5)
        self.status.pack(fill='x')
        self.strip=tk.Canvas(self.window,height=85,background='#f3f5f7',highlightthickness=0)
        self.strip.pack(fill='x')
        self.strip.bind('<Configure>',lambda _:self.draw_strip())
        self.tree=ttk.Treeview(self.window,columns=('cue','owner','reason'),show='headings',selectmode='browse')
        for col,title,width in [('cue','Scene / cue',100),('owner','Destination',200),('reason','Decision / transition',580)]:
            self.tree.heading(col,text=title)
            self.tree.column(col,width=width)
        self.tree.pack(fill='both',expand=True)
        self.tree.bind('<Double-1>',self.open)
        self.tree.bind('<Return>',self.open)
        nav=ttk.Frame(self.window)
        nav.pack()
        self.prev=ttk.Button(nav,text='Previous',command=lambda:self.show(self.page-1))
        self.next=ttk.Button(nav,text='Next',command=lambda:self.show(self.page+1))
        self.prev.pack(side='left')
        self.next.pack(side='left')
        ttk.Button(nav,text='Inspect cue / linked witness',command=self.open).pack(side='left')
        self.family.bind('<<ComboboxSelected>>',self.clear)
        self.direction.bind('<<ComboboxSelected>>',self.clear)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.clear()

    def clear(self,event=None):
        self.worker.invalidate()
        self.data=[]
        self.is_track=self.family.current()==2
        self.filter()
        self.run.state(['!disabled'])
        self.status.configure(text='Inspect retained track decisions by cue.' if self.is_track else
                'Test source face queries against the frozen catalogue in a private copy using the current YAML.')

    def load(self):
        self.clear()
        p=self.parent
        ids=p.pair() if self.direction.current()==0 else p.pair()[::-1]
        track=self.is_track
        task=(partial(timeline,p.reader,p.scan,p.pair()) if track else
              partial(simulate,p.reader.config,p.scan['scope'],ids,('view','hybrid')[self.family.current()],p.revision))
        def read():
            if stamp(p.reader.config.database)!=p.revision:
                raise ValueError('Recognition changed. Refresh the comparison.')
            return task()
        def done(result):
            self.run.state(['!disabled'])
            self.data=result if track else result['rows']
            if track:
                loss=next((r for r in result if r['loss']),None)
                message=f"Entry scene · {len(result)} bodies · "+(f"first loss on a recorded link at cue {loss['cue']}" if loss else
                    'no assignment loss established by a recorded link')
            else:
                first=self.data[0]
                counts=', '.join(f"{k}: {v}" for k,v in result['counts'].items())
                message=(f"{counts} · first query: scene {first['scene']} cue {first['cue']} → {first['state']}. "
                         f"{result['bodies']} source body observations outside this face stage. Verify the complete pair by replay. "
                         +result['advice'])
            self.status.configure(text=message)
            self.filter()
        def failed(error):
            self.run.state(['!disabled'])
            self.status.configure(text=str(error))
        self.run.state(['disabled'])
        self.status.configure(text='Reading evidence…' if track else 'Preparing a private attachment trial…')
        self.worker.request(read,done,failed)

    def filter(self):
        self.filtered=[r for r in self.data if not self.only.get() or
                       (r['loss'] or r.get('blocked',False) if self.is_track else r['state']!='target')]
        self.show(0)

    def show(self,page):
        self.page=page
        self.tree.delete(*self.tree.get_children())
        for i,r in enumerate(self.filtered[page*100:(page+1)*100],page*100):
            self.tree.insert('','end',iid=str(i),values=(f"{r['scene']} / {r['cue']}",r.get('state') or r['owner'] or 'Unassigned',
                ' · '.join(filter(None,(r['reason'],r.get('signal'))))))
        self.draw_strip()
        self.prev.state(['!disabled'] if page else ['disabled'])
        self.next.state(['!disabled'] if (page+1)*100<len(self.filtered) else ['disabled'])

    def draw_strip(self):
        self.strip.delete('all')
        if not self.is_track:
            self.strip.configure(height=1)
            return
        self.strip.configure(height=85)
        rows=self.filtered[self.page*100:(self.page+1)*100]
        if not rows: return
        ids=self.parent.pair()
        width=max(200,self.strip.winfo_width()-90)
        start,end=rows[0]['cue'],rows[-1]['cue']
        for lane,label in enumerate(('A','B','Other / lost')):
            y=16+lane*23
            self.strip.create_text(4,y,text=label,anchor='w')
            self.strip.create_line(78,y,width+78,y,fill='#c1c9d0')
        for i,r in enumerate(rows,self.page*100):
            lane=ids.index(r['owner']) if r['owner'] in ids else 2
            x=78+(r['cue']-start)*width/max(1,end-start)
            y=16+lane*23
            color='#bd3b28' if r['loss'] or r['blocked'] else '#295e85' if r['face'] else '#53775b'
            mark=self.strip.create_oval(x-4,y-4,x+4,y+4,fill=color,outline='')
            def select(event,index=str(i)):
                self.tree.selection_set(index)
                self.tree.see(index)
            self.strip.tag_bind(mark,'<Button-1>',select)
        self.strip.create_text(80,78,text=f'cue {start}',anchor='w')
        self.strip.create_text(width+78,78,text=f'cue {end}',anchor='e')

    def open(self,event=None):
        selection=self.tree.selection()
        if selection:
            r=self.filtered[int(selection[0])]
            refs=[(r['kind'],r['id'])]
            if r.get('refs'):
                refs.append(('body',r['refs'][0]))
            elif r.get('face'):
                refs.append(('face',r['face']))
            self.parent.view(dict(label=r['reason'],refs=refs))

    def close(self):
        self.worker.close()
        self.window.destroy()
        self.parent.fragment_window=None
