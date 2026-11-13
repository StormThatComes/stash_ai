"""Explicit visual exploration: pinned comparisons, cue context and bounded galleries."""
from functools import partial
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import ImageTk
from .inspection_runtime import Worker, check_cancelled
from .inspection_overlays import render, point_info
from .inspection_context import locate

DEFAULT_LAYERS=dict(face_boxes=True,body_boxes=True,landmarks=True,keypoints=True,skeleton=True,masks=False)


class Visual:
    def build_visual(self,right):
        self.pin=None;self.preview_data={};self.layers=dict(DEFAULT_LAYERS);self.context_cue=None
        self.visual_windows=[]
        bar=ttk.Frame(right);bar.grid(row=4,column=0,sticky='ew')
        self.context_mode=tk.StringVar(master=self.root,value='Crop')
        mode=ttk.Combobox(bar,textvariable=self.context_mode,values=['Crop','Full cue'],width=9,state='readonly')
        mode.pack(side='left');mode.bind('<<ComboboxSelected>>',lambda _:self.request_selected())
        if self.previews is None:mode.configure(state='disabled')
        ttk.Button(bar,text='← cue',command=lambda:self.move_cue(-1)).pack(side='left')
        ttk.Button(bar,text='cue →',command=lambda:self.move_cue(1)).pack(side='left')
        ttk.Button(bar,text='Nearby cues',command=self.cue_strip).pack(side='left')
        ttk.Button(bar,text='Pin A → B',command=self.pin_selected).pack(side='left')
        ttk.Button(bar,text='Unpin',command=self.unpin).pack(side='left')
        bar2=ttk.Frame(right);bar2.grid(row=5,column=0,sticky='ew')
        ttk.Button(bar2,text='Gallery',command=self.gallery).pack(side='left')
        ttk.Button(bar2,text='Open observation…',command=self.open_id).pack(side='left')
        ttk.Button(bar2,text='Geometry layers…',command=self.layer_dialog).pack(side='left')
        self.point_label=ttk.Label(right,text='',wraplength=700);self.point_label.grid(row=6,column=0,sticky='w')
        for name,panel in self.images.panels.items():
            panel.canvas.bind('<Motion>',lambda e,n=name:self.inspect_point(n,e))
            panel.canvas.bind('<Double-Button-1>',lambda e,n=name:self.select_shape(n,e))

    def request_selected(self,cue=None):
        if not self.current_row or self.preview_scope is None:return
        row=self.current_row
        self.preview_data.pop('A',None)
        self.context_cue=cue
        full=self.context_mode.get()=='Full cue'
        self.images.show('A',None,'Loading full cue…' if full else 'Loading crop…')
        self.image_worker.request(partial(self.load_reference,row['reader'],row['kind'],row['id'],self.preview_scope,
            full=full,cue=cue,layers=dict(self.layers)),partial(self.display_preview,'A'),partial(self.image_error,'A'))

    def move_cue(self,direction):
        if self.previews is None or not self.current_row:return
        data=self.preview_data.get('A',{}).get('geometry',{})
        current=data.get('cue_index',self.current_row['cue'])
        target=current+direction
        if target<0 or ('cue_count' in data and target>=data['cue_count']):return
        self.context_mode.set('Full cue');self.request_selected(target)

    def pin_selected(self):
        if not self.current_row or 'A' not in self.preview_data:return
        self.pin=dict(row=dict(self.current_row),scope=self.preview_scope,preview=self.preview_data['A'])
        self.reference_worker.invalidate()
        self.images.panels['B'].configure(text='Pinned comparison · user selected')
        self.display_preview('B',self.pin['preview'])

    def unpin(self):
        self.pin=None;self.images.panels['B'].configure(text='Recorded reference');self.show_reference()

    def visual_preview(self,panel,result):
        self.preview_data[panel]=result
        if panel=='A' and 'cue_index' in result['geometry']:
            g=result['geometry'];self.point_label.configure(text=f"Cue {g['cue_index']} / {g['cue_count']-1} · {g['cue_seconds']:g} s · double-click a box to inspect")

    def inspect_point(self,panel,event):
        data=self.preview_data.get(panel)
        if not data or data['image'][0] is None:return
        p=self.images.panels[panel];x=p.canvas.canvasx(event.x)/p.zoom;y=p.canvas.canvasy(event.y)/p.zoom
        text=point_info(data['geometry'],data['image'][0].size,x,y)
        if text:self.point_label.configure(text=text)

    def select_shape(self,panel,event):
        data=self.preview_data.get(panel)
        if not data or data['image'][0] is None:return
        bounds=data['geometry'].get('bounds')
        if not bounds:return
        p=self.images.panels[panel];im=data['image'][0]
        x=bounds[0]+p.canvas.canvasx(event.x)/p.zoom*(bounds[2]-bounds[0])/im.width
        y=bounds[1]+p.canvas.canvasy(event.y)/p.zoom*(bounds[3]-bounds[1])/im.height
        choices=[]
        for shape in data['geometry']['shapes']:
            if not shape.get('id'):continue
            g=shape['geometry']
            if shape['kind']=='face' and (not isinstance(g,list) or len(g)<4):continue
            if shape['kind']=='body' and not isinstance(g,dict):continue
            box=[g[0],g[1],g[0]+g[2],g[1]+g[3]] if shape['kind']=='face' else g.get('bounds')
            from .inspection_overlays import _numbers
            if _numbers(box,4) and box[0]<=x<box[2] and box[1]<=y<box[3]:choices.append(((box[2]-box[0])*(box[3]-box[1]),shape))
        if choices:
            shape=min(choices,key=lambda v:v[0])[1];self.locate_observation(shape['kind'],shape['id'])

    def locate_observation(self,kind,address):
        if self.preview_scope:
            self.worker.request(partial(locate,self.reader.config,kind,address,self.preview_scope),self.open_row,self.error)

    def open_id(self):
        value=simpledialog.askstring('Open observation','face:OBSERVATION_ID or body:OBSERVATION_ID',parent=self.root)
        if value and ':' in value:
            kind,address=value.split(':',1);self.locate_observation(kind.strip(),address.strip())

    def open_row(self,row):
        self.detail_reader=row['reader']
        if hasattr(self,'branches'):
            parent=self.listing.insert('','end',text='Context selection',open=True)
            item=self.listing.insert(parent,'end',text=row['id'],values=(row['scene'],row['cue'],row['kind'],row['state']))
        else:
            item=self.listing.insert('','end',values=(row['scene'],row['cue'],row['kind']))
        self.rows[item]=row;self.listing.selection_set(item);self.listing.see(item)

    def layer_dialog(self):
        window=tk.Toplevel(self.root);window.title('Geometry layers')
        variables={}
        for key,value in self.layers.items():
            v=variables[key]=tk.BooleanVar(master=window,value=value)
            ttk.Checkbutton(window,text=key.replace('_',' ').capitalize(),variable=v).pack(anchor='w')
        ttk.Label(window,text='Only retained geometry/masks can be shown.\nPoint confidence may not have been retained.').pack()
        def apply():
            self.layers={k:v.get() for k,v in variables.items()};window.destroy()
            for panel,data in list(self.preview_data.items()):
                def repaint(data=data,layers=dict(self.layers)):
                    result=dict(data);image,caption=data['image'];annotated,label=render(image,data['geometry'],layers)
                    result['annotated']=annotated,caption+' · '+label
                    return result
                worker=self.image_worker if panel=='A' else self.reference_worker
                worker.request(repaint,partial(self.display_preview,panel),partial(self.image_error,panel))
        ttk.Button(window,text='Apply',command=apply).pack()

    def gallery(self):
        selection=self.listing.selection();item=selection[0] if selection else None
        if hasattr(self,'branches'):
            branch=self.branches.get(item) or self.branches.get(self.listing.parent(item)) if item else None
            if not branch or branch['level']=='states':
                self.status.set('Select an observation state or search-results branch for Gallery.');return
            reader=branch['reader']
            args=dict(kind=branch['kind'],state=branch['state'])
            if not branch['unassigned']:args['accepted_only']=True
        else:reader=self.reader;args=dict(self.filters)
        window=Gallery(self,reader,args);self.visual_windows.append(window)

    def cue_strip(self):
        if not self.current_row or self.previews is None:return
        window=Gallery(self,None,{},cues=True);self.visual_windows.append(window)


class Gallery:
    SIZE=12
    def __init__(self,host,reader,args,*,cues=False):
        self.host,self.reader,self.args,self.cues=host,reader,args,cues
        self.layers=dict(host.layers)
        self.row=dict(host.current_row) if host.current_row else None
        self.scope=host.preview_scope or getattr(host,'scan',{}).get('scope')
        self.center=host.preview_data.get('A',{}).get('geometry',{}).get('cue_index',self.row['cue'] if self.row else 0)
        self.count=host.preview_data.get('A',{}).get('geometry',{}).get('cue_count')
        self.page=0;self.photos=[];self.closed=False
        self.window=tk.Toplevel(host.root);self.window.title('Nearby VTT cues' if cues else 'Observation gallery')
        self.grid=ttk.Frame(self.window);self.grid.pack(fill='both',expand=True)
        bar=ttk.Frame(self.window);bar.pack(fill='x')
        self.previous=ttk.Button(bar,text='Previous',command=lambda:self.move(-1));self.previous.pack(side='left')
        self.next=ttk.Button(bar,text='Next',command=lambda:self.move(1));self.next.pack(side='left')
        self.status=ttk.Label(bar,text='');self.status.pack(side='left')
        self.worker=Worker('inspector-gallery',resource=host.previews.fork if host.previews else None)
        self.window.protocol('WM_DELETE_WINDOW',self.close);self.timer=self.window.after(30,self.poll);self.load()

    def move(self,direction):
        self.page=max(0,self.page+direction);self.load()

    def load(self):
        page=self.page;self.status.configure(text='Loading thumbnails…')
        self.previous.state(['disabled'] if page==0 else ['!disabled']);self.next.state(['disabled'])
        def work():
            if self.cues:
                if self.count is None:
                    preview=self.host.load_reference(self.row['reader'],self.row['kind'],self.row['id'],self.scope,full=True)
                    self.count=preview['geometry'].get('cue_count')
                    if self.count is None:raise ValueError('VTT context unavailable')
                start=max(0,self.center-2)+page*5
                stop=min(start+5,self.count) if self.count is not None else self.center+3
                rows=[dict(self.row,preview_cue=i) for i in range(start,stop)];more=self.count is not None and stop<self.count
            else:
                rows,more=self.reader.page(offset=page*self.SIZE,**self.args)
                more=more or len(rows)>self.SIZE;rows=rows[:self.SIZE]
            result=[]
            for row in rows:
                check_cancelled()
                reader=row.get('reader') or (self.reader.owner(row) if hasattr(self.reader,'owner') else self.reader)
                row=dict(row,reader=reader)
                scope=self.scope or reader.detail(row['kind'],row['id'],pixels=False)['scope']
                preview=self.host.load_reference(reader,row['kind'],row['id'],scope,full=self.cues,cue=row.get('preview_cue'),layers=self.layers)
                image,caption=preview['annotated'];thumb=None
                if image is not None:thumb=image.copy();thumb.thumbnail((150,130))
                result.append((row,thumb,caption))
            return result,more
        self.worker.request(work,self.show,lambda e:self.status.configure(text=f'Gallery failed ({type(e).__name__}); change page to retry.'))

    def show(self,result):
        rows,more=result
        for widget in self.grid.winfo_children():widget.destroy()
        self.photos=[]
        for index,(row,image,caption) in enumerate(rows):
            frame=ttk.Frame(self.grid,padding=4);frame.grid(row=index//4,column=index%4,sticky='nsew')
            photo=ImageTk.PhotoImage(image,master=self.window) if image is not None else None
            self.photos.append(photo)
            command=(lambda i=row['preview_cue']:self.jump_cue(i)) if self.cues else (lambda r=row:self.host.open_row(r))
            ttk.Button(frame,image=photo,text='Crop unavailable' if photo is None else '',command=command).pack()
            if image is None:ttk.Label(frame,text=caption,wraplength=150).pack()
            ttk.Label(frame,text=f"Scene {row['scene']} · cue {row.get('preview_cue',row['cue'])}\n{row['kind']} · {row.get('state','')}",wraplength=150).pack()
        self.next.state(['!disabled'] if more else ['disabled']);self.status.configure(text=f'Page {self.page+1} · {len(rows)} thumbnails')

    def jump_cue(self,index):
        if not self.host.current_row or self.host.current_row['id']!=self.row['id']:
            self.host.status.set('Select the original observation before navigating its cue strip.');return
        self.host.context_mode.set('Full cue');self.host.request_selected(index)

    def poll(self):
        if not self.closed:self.worker.poll();self.timer=self.window.after(30,self.poll)

    def close(self):
        if self.closed:return
        self.closed=True;self.window.after_cancel(self.timer);self.worker.close();self.window.destroy()
