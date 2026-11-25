"""Native animated graph; all reads/layout work stay outside Tk callbacks."""
from functools import partial
import json
import tkinter as tk
from tkinter import ttk
from .inspection_runtime import Worker, stamp
from .inspection_graph import build, PAGE_SIZE
from .inspection_graph_layout import initial, step, project
from .inspection_context import locate

COLORS = dict(anchor='#2563eb',auxiliary='#d97706',attachment='#7c3aed',unknown='#64748b',reference='#94a3b8')


class GraphWindow:
    def __init__(self, host):
        self.host = host
        self.reader = host.reader
        self.window = tk.Toplevel(host.root)
        self.window.title('Performer observation graph')
        self.window.geometry('1150x800')
        self.worker, self.physics = Worker('performer-graph'), Worker('graph-layout')
        self.closed = False
        self.groups=set();self.nodes={};self.display=dict(nodes=[],edges=[])
        self.graph = None; self.positions = {}; self.pinned = set(); self.expanded = set()
        self.offset=0; self.selected=None; self.dragging=None; self.drag_start=None
        self.scale=1.;self.origin=[550.,330.];self.paused=False;self.iterations=0
        self.items={};self.lines=[];self.item_nodes={};self.item_edges={}
        self.filters={k:v for k,v in host.filters.items() if k in ('scene','kind','state')}
        bar=ttk.Frame(self.window,padding=5);bar.pack(fill='x')
        for label,command in [('Refresh',self.refresh),('Previous',lambda:self.page(-1)),('Next',lambda:self.page(1)),
                ('Pause / resume',self.pause),('Fit',self.fit),('Reset layout',self.reset),
                ('Expand groups',self.expand_all),('Collapse groups',self.collapse_all)]:
            ttk.Button(bar,text=label,command=command).pack(side='left')
        ttk.Label(self.window,text='Drag: move/pin node or pan background · Right-click: unpin · Double-click: expand group / inspect observation\n'
            'Blue: anchor · Amber: auxiliary · Purple: attachment · Square: body · Diamond: View · Small dot: decision\n'
            'Distances are layout only. Groups are visual containers. Missing links may reflect missing evidence.').pack(anchor='w')
        self.canvas=tk.Canvas(self.window,background='#f8fafc',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.info=tk.Text(self.window,height=6,wrap='word',state='disabled');self.info.pack(fill='x')
        self.status=ttk.Label(self.window,wraplength=1100);self.status.pack(fill='x')
        self.canvas.bind('<ButtonPress-1>',self.press);self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.release);self.canvas.bind('<Button-3>',self.unpin)
        self.canvas.bind('<Double-Button-1>',self.activate);self.canvas.bind('<Motion>',self.hover)
        self.canvas.bind('<MouseWheel>',lambda e:self.zoom(e,1.15 if e.delta>0 else 1/1.15))
        self.canvas.bind('<Button-4>',lambda e:self.zoom(e,1.15))
        self.canvas.bind('<Button-5>',lambda e:self.zoom(e,1/1.15))
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.timer=self.window.after(33,self.tick)
        self.refresh()

    def refresh(self):
        self.physics.invalidate();self.iterations=200
        self.status.configure(text='Loading retained evidence…')
        self.worker.request(partial(build,self.reader,offset=self.offset,**self.filters),self.loaded,self.error)

    def page(self, direction):
        if direction>0 and (not self.graph or not self.graph['more']):return
        self.offset=max(0,self.offset+direction*PAGE_SIZE);self.refresh()

    def loaded(self, graph):
        self.graph=graph
        self.status.configure(text=f"Current observations {self.offset+1 if graph['total'] else 0}–{min(self.offset+PAGE_SIZE,graph['total'])} / {graph['total']} · "
            f"{graph['omitted']} context nodes omitted · Retained references may be incomplete")
        self.rebuild()

    def rebuild(self):
        if self.graph is None:return
        self.physics.invalidate()
        self.display,self.groups=project(self.graph,self.expanded)
        self.nodes={n['id']:n for n in self.display['nodes']}
        self.positions=initial(self.nodes,self.positions)
        self.pinned.intersection_update(self.positions)
        self.iterations=0
        self.canvas.delete('all');self.items={};self.lines=[];self.item_nodes={};self.item_edges={}
        for edge in self.display['edges']:
            line=self.canvas.create_line(0,0,0,0,fill='#94a3b8',arrow='last',dash=(4,3) if edge['relation'] in ('similarity','rival') else ())
            self.lines.append((line,edge));self.item_edges[line]=edge
        for identifier,n in self.nodes.items():
            color=COLORS.get(n.get('state'),'#059669' if n['type']=='view' else '#334155')
            if n['type']=='group':
                item=self.canvas.create_rectangle(0,0,0,0,fill='#e2e8f0',outline='#475569',width=2)
            elif n['type']=='view':
                item=self.canvas.create_polygon(0,0,0,0,0,0,0,0,fill=color)
            elif n.get('kind')=='body':item=self.canvas.create_rectangle(0,0,0,0,fill=color)
            else:item=self.canvas.create_oval(0,0,0,0,fill=color,outline='')
            label=(f"{n.get('label','Group')[:32]} · {len(n['members'])}" if n['type']=='group' else
                   'View' if n['type']=='view' else '' if n['type']=='decision' else f"{n.get('scene','?')}:{n.get('cue','?')}")
            text=self.canvas.create_text(0,0,text=label,fill='#0f172a',anchor='n',font=('TkDefaultFont',9))
            self.items[identifier]=(item,text);self.item_nodes[item]=identifier;self.item_nodes[text]=identifier
        self.draw()

    def draw(self):
        def point(identifier):
            x,y=self.positions[identifier];return self.origin[0]+x*self.scale,self.origin[1]+y*self.scale
        for item,e in self.lines:self.canvas.coords(item,*point(e['source']),*point(e['target']))
        for identifier,(item,text) in self.items.items():
            n=self.nodes[identifier];x,y=point(identifier)
            radius=(18 if n['type']=='group' else 12 if n['type']=='view' else 4 if n['type']=='decision' else 8)*min(self.scale,2)
            if n['type']=='view':self.canvas.coords(item,x,y-radius,x+radius,y,x,y+radius,x-radius,y)
            else:self.canvas.coords(item,x-radius,y-radius,x+radius,y+radius)
            self.canvas.coords(text,x,y+radius+3)
            self.canvas.itemconfigure(item,width=3 if identifier==self.selected or identifier in self.pinned else 1)

    def tick(self):
        if self.closed:return
        self.worker.poll();self.physics.poll()
        if self.graph and not self.paused and self.iterations<200 and self.physics.future is None and not self.dragging:
            self.physics.request(partial(step,dict(self.positions),self.display['edges'],set(self.pinned)),self.moved,self.error)
        self.timer=self.window.after(33,self.tick)

    def moved(self, result):
        self.positions,movement=result;self.iterations+=1
        if movement<.08:self.iterations=200
        self.draw()

    def at(self,event):
        items=self.canvas.find_overlapping(event.x-4,event.y-4,event.x+4,event.y+4)
        return next((self.item_nodes[i] for i in reversed(items) if i in self.item_nodes),None)

    def press(self,event):
        self.physics.invalidate();self.dragging=self.at(event);self.drag_start=(event.x,event.y)
        if self.dragging:
            self.selected=self.dragging;self.describe(self.nodes[self.selected]);self.draw()
        else:
            items=self.canvas.find_overlapping(event.x-3,event.y-3,event.x+3,event.y+3)
            edge=next((self.item_edges[i] for i in items if i in self.item_edges),None)
            if edge:self.describe(edge)

    def drag(self,event):
        if self.drag_start is None:return
        dx,dy=event.x-self.drag_start[0],event.y-self.drag_start[1]
        if self.dragging:
            x,y=self.positions[self.dragging]
            self.positions[self.dragging]=(x+dx/self.scale,y+dy/self.scale);self.pinned.add(self.dragging)
        else:self.origin[0]+=dx;self.origin[1]+=dy
        self.drag_start=(event.x,event.y);self.draw()

    def release(self,event):
        self.dragging=None;self.drag_start=None;self.iterations=0

    def unpin(self,event):
        self.pinned.discard(self.at(event));self.iterations=0;self.draw()

    def activate(self,event):
        identifier=self.at(event)
        if not identifier:return
        n=self.nodes[identifier]
        if n['type']=='group':self.expanded.add(identifier);self.rebuild()
        elif n['type']=='observation':self.inspect(n)

    def inspect(self,n):
        if self.graph['revision']!=stamp(self.reader.config.database):
            self.error(ValueError('Recognition changed. Refresh before navigating.'));return
        if n.get('availability')=='unavailable_in_scope':
            self.describe(n);return
        if n.get('current_member'):
            self.worker.request(partial(locate,self.reader.config,n['kind'],n['observation'],self.graph['person']['scope']),
                                self.host.open_row,self.error)
        else:
            # Historical references remain previewable without masquerading as current members.
            self.host.reference_worker.request(partial(self.host.load_reference,self.reader,n['kind'],n['observation'],
                self.graph['person']['scope']),partial(self.host.display_preview,'B'),self.error)

    def hover(self,event):
        identifier=self.at(event)
        if identifier:self.status.configure(text=identifier+' · '+self.nodes[identifier].get('provenance',''))

    def describe(self,value):
        self.info.configure(state='normal');self.info.delete('1.0','end')
        self.info.insert('1.0',json.dumps(value,indent=2,ensure_ascii=False,default=str));self.info.configure(state='disabled')

    def zoom(self,event,factor):
        old=self.scale;self.scale=max(.15,min(4.,old*factor));ratio=self.scale/old
        self.origin=[event.x+(self.origin[0]-event.x)*ratio,event.y+(self.origin[1]-event.y)*ratio];self.draw()

    def fit(self):
        if not self.positions:return
        xs,ys=zip(*self.positions.values());w=max(200,self.canvas.winfo_width());h=max(200,self.canvas.winfo_height())
        self.scale=min(2.,(w-100)/max(100,max(xs)-min(xs)),(h-100)/max(100,max(ys)-min(ys)))
        self.origin=[w/2-(max(xs)+min(xs))/2*self.scale,h/2-(max(ys)+min(ys))/2*self.scale];self.draw()

    def pause(self):
        self.paused=not self.paused;self.physics.invalidate();self.iterations=0

    def reset(self):
        self.physics.invalidate();self.positions=initial(self.nodes);self.pinned.clear();self.iterations=0;self.fit()

    def expand_all(self):self.expanded=set(self.groups);self.rebuild()
    def collapse_all(self):self.expanded.clear();self.rebuild()
    def error(self,error):self.status.configure(text=str(error))

    def close(self):
        if self.closed:return
        self.closed=True;self.window.after_cancel(self.timer);self.worker.close();self.physics.close();self.window.destroy()
        if self in self.host.visual_windows:self.host.visual_windows.remove(self)
