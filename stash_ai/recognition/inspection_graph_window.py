"""Native animated graph; all reads/layout work stay outside Tk callbacks."""
from functools import partial
import json
from queue import Queue, Empty, Full
from PIL import ImageTk
import tkinter as tk
from tkinter import ttk, simpledialog
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
        self.loading=False;self.progress_events=Queue(maxsize=1);self.photo_timer=None
        self.pending_graph=None;self.pending_thumbnails=None
        self.thumbnail_images={};self.thumbnail_captions={};self.thumbnail_items={}
        self.show_thumbnails=tk.BooleanVar(master=self.window,value=True)
        self.groups=set();self.nodes={};self.display=dict(nodes=[],edges=[])
        self.graph = None; self.positions = {}; self.pinned = set(); self.expanded = set()
        self.offset=0; self.selected=None; self.dragging=None; self.drag_start=None
        self.scale=1.;self.origin=[550.,330.];self.paused=False;self.iterations=0
        self.items={};self.lines=[];self.item_nodes={};self.item_edges={}
        self.highlight=set();self.similarity_edges=[];self.merge_window=None;self.visual_windows=[]
        self.filters={k:v for k,v in host.filters.items() if k in ('scene','kind','state')}
        bar=ttk.Frame(self.window,padding=5);bar.pack(fill='x')
        for label,command in [('Refresh',self.refresh),('Previous',lambda:self.page(-1)),('Next',lambda:self.page(1)),
                ('Pause / resume',self.pause),('Fit',self.fit),('Reset layout',self.reset),
                ('Expand groups',self.expand_all),('Collapse groups',self.collapse_all)]:
            ttk.Button(bar,text=label,command=command).pack(side='left')
        self.action_bars=[bar]
        diagnostics=ttk.Frame(self.window,padding=5);diagnostics.pack(fill='x')
        for label,command in [('Why here?',lambda:self.dependencies(False)),
                ('Uses this evidence',lambda:self.dependencies(True)),('Weak connections',self.weak_connections),
                ('Compare similarity…',self.compare_similarity),('Clear similarity',self.clear_similarity),
                ('Compare performer…',self.compare_performer)]:
            ttk.Button(diagnostics,text=label,command=command).pack(side='left')
        self.action_bars.append(diagnostics)
        ttk.Checkbutton(diagnostics,text='Thumbnails',variable=self.show_thumbnails,command=self.rebuild).pack(side='left')
        ttk.Label(self.window,text='Drag: move/pin node or pan background · Right-click: unpin · Double-click: expand group / inspect observation\n'
            'Blue: anchor · Amber: auxiliary · Purple: attachment · Face/body labels on thumbnails · Diamond: View · Small dot: decision\n'
            'Distances are layout only. Groups are visual containers. Missing links may reflect missing evidence.').pack(anchor='w')
        self.loading_panel=ttk.Frame(self.window,padding=30)
        self.loading_label=ttk.Label(self.loading_panel,text='Preparing graph…')
        self.loading_label.pack(pady=8)
        self.progress=ttk.Progressbar(self.loading_panel,length=450,mode='indeterminate')
        self.progress.pack(fill='x',pady=8)
        ttk.Button(self.loading_panel,text='Cancel loading',command=self.cancel_loading).pack(pady=8)
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
        self.stop_loading()
        self.worker.invalidate();self.physics.invalidate();self.iterations=200
        self.loading=True;self.progress_events=Queue(maxsize=1)
        self.set_controls(False)
        self.canvas.pack_forget()
        self.loading_panel.pack(fill='both',expand=True,before=self.info)
        self.loading_label.configure(text='Preparing graph evidence…')
        self.progress.configure(mode='indeterminate');self.progress.start(15)
        self.status.configure(text='Preparing graph and retained observation thumbnails…')
        self.worker.request(partial(build,self.reader,offset=self.offset,**self.filters),self.loaded,self.load_error)

    def set_controls(self, enabled):
        for bar in self.action_bars:
            for widget in bar.winfo_children():
                widget.state(['!disabled'] if enabled else ['disabled'])

    def page(self, direction):
        if self.loading:return
        if direction>0 and (not self.graph or not self.graph['more']):return
        self.offset=max(0,self.offset+direction*PAGE_SIZE);self.refresh()

    def loaded(self, graph):
        from .inspection_graph_thumbnails import load
        self.pending_graph=graph
        self.progress.stop();self.progress.configure(mode='determinate',maximum=1,value=0)
        self.loading_label.configure(text='Loading retained thumbnails…')
        events=self.progress_events=Queue(maxsize=1)
        def progress(done,total):
            # Only the newest progress matters. This queue belongs to this load;
            # cancelled workers cannot publish progress into a replacement load.
            try:events.get_nowait()
            except Empty:pass
            try:events.put_nowait((done,total))
            except Full:pass
        self.worker.request(partial(load,self.reader.config,graph,progress),self.thumbnails_loaded,self.load_error)

    def thumbnails_loaded(self,result):
        self.pending_thumbnails=result
        self.photo_queue=iter(result['images'].items());self.pending_photos={}
        self.progress_events=Queue(maxsize=1)
        self.progress.configure(maximum=max(1,len(result['images'])),value=0)
        self.prepare_photos()

    def prepare_photos(self):
        self.photo_timer=None
        if self.closed or not self.loading:return
        # Tk images belong to the UI thread. Short batches keep Cancel/Close live.
        try:
            for _ in range(12):
                identifier,pixels=next(self.photo_queue)
                self.pending_photos[identifier]=ImageTk.PhotoImage(pixels,master=self.canvas)
        except StopIteration:
            self.finish_loading();return
        except Exception as error:
            self.load_error(error);return
        done=len(self.pending_photos);total=len(self.pending_thumbnails['images'])
        self.loading_label.configure(text=f'Preparing thumbnails {done} / {total}…')
        self.progress.configure(value=done)
        self.photo_timer=self.window.after(1,self.prepare_photos)

    def finish_loading(self):
        if stamp(self.reader.config.database)!=self.pending_graph['revision']:
            self.load_error(ValueError('Recognition changed while preparing thumbnails. Refresh the graph.'));return
        self.highlight.clear();self.similarity_edges=[]
        self.graph=self.pending_graph
        self.thumbnail_images=self.pending_photos
        self.thumbnail_captions=self.pending_thumbnails['captions']
        missing=self.pending_thumbnails['missing'];total=self.pending_thumbnails['total']
        self.stop_loading()
        self.rebuild()
        self.status.configure(text=f"Current observations {self.offset+1 if self.graph['total'] else 0}–{min(self.offset+PAGE_SIZE,self.graph['total'])} / {self.graph['total']} · "
            f"Thumbnails {total-missing}/{total} · {missing} unavailable · {self.graph['omitted']} context nodes omitted")

    def stop_loading(self):
        if self.photo_timer is not None:
            self.window.after_cancel(self.photo_timer);self.photo_timer=None
        self.progress.stop();self.loading=False
        self.pending_graph=None;self.pending_thumbnails=None;self.pending_photos={};self.photo_queue=iter(())
        self.loading_panel.pack_forget()
        self.canvas.pack(fill='both',expand=True,before=self.info)
        self.set_controls(True)

    def cancel_loading(self):
        self.worker.invalidate();self.physics.invalidate()
        if self.graph is not None:self.offset=self.graph['offset']
        self.stop_loading()
        self.status.configure(text='Loading cancelled. Refresh to retry.')

    def load_error(self,error):
        if self.graph is not None:self.offset=self.graph['offset']
        self.stop_loading();self.error(error)

    def thumbnail(self,node):
        if not self.show_thumbnails.get():return None
        if node['type']=='group':
            # Explicitly a sample, never a group portrait or a new medoid.
            identifier=next((key for key in node['members'] if key in self.thumbnail_images),None)
        else:identifier=node['id']
        return self.thumbnail_images.get(identifier)

    def rebuild(self):
        if self.graph is None or self.loading:return
        self.physics.invalidate()
        projected=dict(self.graph,edges=self.graph['edges']+self.similarity_edges)
        self.display,self.groups=project(projected,self.expanded)
        self.nodes={n['id']:n for n in self.display['nodes']}
        self.positions=initial(self.nodes,self.positions)
        self.pinned.intersection_update(self.positions)
        self.iterations=0
        self.canvas.delete('all');self.items={};self.lines=[];self.item_nodes={};self.item_edges={};self.thumbnail_items={}
        for edge in self.display['edges']:
            line=self.canvas.create_line(0,0,0,0,fill='#94a3b8',arrow='last',dash=(4,3) if edge['relation'] in ('similarity','rival') else ())
            self.lines.append((line,edge));self.item_edges[line]=edge
        for identifier,n in self.nodes.items():
            color=COLORS.get(n.get('state'),'#059669' if n['type']=='view' else '#334155')
            thumbnail=self.thumbnail(n)
            if thumbnail is not None:
                item=self.canvas.create_rectangle(0,0,0,0,fill=color,outline=color,width=3)
            elif n['type']=='group':
                item=self.canvas.create_rectangle(0,0,0,0,fill='#e2e8f0',outline='#475569',width=2)
            elif n['type']=='view':
                item=self.canvas.create_polygon(0,0,0,0,0,0,0,0,fill=color)
            elif n.get('kind')=='body':item=self.canvas.create_rectangle(0,0,0,0,fill=color)
            else:item=self.canvas.create_oval(0,0,0,0,fill=color,outline='')
            if thumbnail is not None:
                photo=self.canvas.create_image(0,0,image=thumbnail)
                self.thumbnail_items[identifier]=photo;self.item_nodes[photo]=identifier
            label=(f"{n.get('label','Group')[:32]} · {len(n['members'])}" + (' · sample' if thumbnail else '') if n['type']=='group' else
                   'View' if n['type']=='view' else '' if n['type']=='decision' else f"{n.get('kind','?')} {n.get('scene','?')}:{n.get('cue','?')}")
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
            thumbnail=self.thumbnail(n)
            if thumbnail is not None:
                # Keep readable screen-sized thumbnails while zoom moves nodes.
                # No pixel decoding/resizing occurs during physics or wheel events.
                rx,ry=thumbnail.width()/2+3,thumbnail.height()/2+3
                self.canvas.coords(item,x-rx,y-ry,x+rx,y+ry)
                self.canvas.coords(self.thumbnail_items[identifier],x,y)
                radius=ry
            elif n['type']=='view':self.canvas.coords(item,x,y-radius,x+radius,y,x,y+radius,x-radius,y)
            else:self.canvas.coords(item,x-radius,y-radius,x+radius,y+radius)
            self.canvas.coords(text,x,y+radius+3)
            self.canvas.itemconfigure(item,width=4 if identifier==self.selected or identifier in self.pinned or identifier in self.highlight else 3 if thumbnail is not None else 1,
                outline='#dc2626' if identifier in self.highlight else COLORS.get(n.get('state'),'#334155'))

    def tick(self):
        if self.closed:return
        self.worker.poll();self.physics.poll()
        if self.loading:
            try:
                done,total=self.progress_events.get_nowait()
                self.loading_label.configure(text=f'Loading thumbnails {done} / {total}…')
                self.progress.configure(maximum=max(1,total),value=done)
            except Empty:pass
        if self.graph and not self.loading and not self.paused and self.iterations<200 and self.physics.future is None and not self.dragging:
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

    def dependencies(self, downstream):
        if not self.graph or not self.selected:return
        from .inspection_graph_diagnostics import related
        node=self.nodes.get(self.selected,{})
        selected=node.get('members',[self.selected])
        highlights=set()
        for identifier in selected:highlights.update(related(self.graph,identifier,downstream=downstream))
        self.highlight=highlights;self.expand_all()
        self.describe(dict(explanation='Recorded downstream dependencies; alternate routes require replay.' if downstream else
            'Recorded upstream evidence; missing links do not prove an independent origin.',nodes=sorted(highlights),
            coverage='Current graph page and retained references only'))

    def weak_connections(self):
        if not self.graph:return
        from .inspection_graph_diagnostics import bridges
        found=bridges(self.graph)
        self.highlight={n for pair in found for n in pair};self.expand_all()
        self.describe(dict(explanation='Structural bridges in this partial graph, not evidence of an incorrect assignment.',bridges=found))

    def compare_similarity(self):
        if not self.graph:return
        from .inspection_graph_diagnostics import similarities
        threshold=simpledialog.askfloat('Exploratory similarity','Minimum cosine similarity (same modality, displayed page only)',
                                       parent=self.window,initialvalue=.7,minvalue=-1,maxvalue=1)
        if threshold is not None:
            self.worker.request(partial(similarities,self.reader,self.graph,threshold=threshold),self.show_similarities,self.error)

    def show_similarities(self,result):
        self.similarity_edges=result['edges'];self.expand_all()
        self.describe(dict(explanation='New same-modality comparisons. These edges do not explain historical assignments or create Views.',
                           edges=len(self.similarity_edges),missing_embeddings=result['missing']))

    def clear_similarity(self):
        self.similarity_edges=[]
        if self.graph:self.rebuild()

    def compare_performer(self):
        if not self.graph:return
        if self.merge_window is not None:self.merge_window.window.lift();return
        from .scene_inspection import SceneReader
        from .inspection_pair_evidence import find_identity
        value=simpledialog.askstring('Compare performer','Performer ID or identity UUID in the same scope',parent=self.window)
        if not value:return
        scene=next((n['scene'] for n in self.graph['nodes'] if n.get('current_member')),None)
        if scene is None:return
        reader=SceneReader(self.reader.config,scene,scope=self.graph['person']['scope'])
        def read():
            catalogue=reader.catalogue()
            other=find_identity(reader,catalogue['scan']['scope'],value.strip())
            if other['identity']==self.graph['person']['id']:raise ValueError('Choose a different performer.')
            return reader,catalogue['scan'],other
        self.worker.request(read,self.show_comparison,self.error)

    def show_comparison(self,result):
        from types import SimpleNamespace
        from .inspection_merge_window import MergeWindow
        reader,scan,other=result
        owner=self.graph['person']
        # The existing comparison expects a scene context for tracking/replay.
        host=SimpleNamespace(root=self.window,reader=reader,scan=scan,visual_windows=self.visual_windows,merge_window=None)
        window=MergeWindow(host,[dict(identity=owner['id'],performer=owner['performer']),other])
        host.merge_window=window;self.visual_windows.append(window)
        self.merge_window=window
        original=window.close
        def close():
            original();self.merge_window=None
        window.close=close
        window.window.protocol('WM_DELETE_WINDOW',close)

    def hover(self,event):
        identifier=self.at(event)
        if identifier:
            caption=self.thumbnail_captions.get(identifier,'')
            self.status.configure(text=identifier+' · '+self.nodes[identifier].get('provenance','')+' · '+caption)

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
        for window in list(self.visual_windows):window.close()
        self.stop_loading()
        self.thumbnail_images.clear()
        self.closed=True;self.window.after_cancel(self.timer);self.worker.close();self.physics.close();self.window.destroy()
        if self in self.host.visual_windows:self.host.visual_windows.remove(self)
