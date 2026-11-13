"""Independent or synchronized viewport navigation for inspector images."""
import math
from tkinter import ttk
from .tune_images import ImagePair, ImagePanel


class InspectorImages(ImagePair):
    def __init__(self,parent):
        ttk.Frame.__init__(self,parent)
        self.rowconfigure(0,weight=1)
        self.zoom=1.;self.zooms={'A':1.,'B':1.};self.synchronized=False
        self.panels={name:ImagePanel(self,name,lambda axis,*args,n=name:self.pan_one(n,axis,*args)) for name in ('A','B')}
        self.compare(True)
        for name,panel in self.panels.items():
            canvas=panel.canvas
            canvas.bind('<MouseWheel>',lambda e,n=name:self.wheel(n,e))
            canvas.bind('<Button-4>',lambda e,n=name:self.wheel(n,e,1))
            canvas.bind('<Button-5>',lambda e,n=name:self.wheel(n,e,-1))
            canvas.bind('<ButtonPress-1>',lambda e,n=name:self.panels[n].canvas.scan_mark(e.x,e.y))
            canvas.bind('<B1-Motion>',lambda e,n=name:self.drag(n,e))

    def layout(self):
        if not hasattr(self,'names'):return
        sizes={n:(math.ceil(p.image.width*self.zooms[n]),math.ceil(p.image.height*self.zooms[n])) if p.image else (0,0) for n,p in self.panels.items()}
        common=tuple(max(s[i] for s in sizes.values()) for i in (0,1))
        for name,panel in self.panels.items():
            panel.zoom=self.zooms[name]
            panel.canvas.configure(scrollregion=(0,0,*(common if self.synchronized else sizes[name])))
            panel.refresh()

    def align(self):
        if self.synchronized:self.sync('A')

    def sync(self,name):
        source=self.panels[name].canvas
        for other,panel in self.panels.items():
            if other!=name:
                panel.canvas.xview_moveto(source.xview()[0]);panel.canvas.yview_moveto(source.yview()[0]);panel.refresh()

    def pan_one(self,name,axis,*args):
        panel=self.panels[name]
        getattr(panel.canvas,axis+'view')(*args);panel.refresh()
        if self.synchronized:self.sync(name)

    def drag(self,name,event):
        self.panels[name].canvas.scan_dragto(event.x,event.y,gain=1)
        self.panels[name].refresh()
        if self.synchronized:self.sync(name)

    def wheel(self,name,event,direction=None):
        self.set_zoom(name,self.zooms[name]*(1.25 if (direction if direction is not None else event.delta)>0 else .8),event.x,event.y)
        return 'break'

    def set_zoom(self,name,value,x=0,y=0):
        panel=self.panels[name];canvas=panel.canvas
        source=(canvas.canvasx(x)/self.zooms[name],canvas.canvasy(y)/self.zooms[name])
        value=max(.05,min(8.,float(value)))
        for target in self.names if self.synchronized else (name,):self.zooms[target]=value
        self.layout()
        region=tuple(map(float,canvas.cget('scrollregion').split()))
        if region[2]>0:canvas.xview_moveto(max(0,source[0]*value-x)/region[2])
        if region[3]>0:canvas.yview_moveto(max(0,source[1]*value-y)/region[3])
        panel.refresh()
        if self.synchronized:self.sync(name)

    def fit(self,name):
        panel=self.panels[name]
        if panel.image:
            self.set_zoom(name,min(max(1,panel.canvas.winfo_width())/panel.image.width,max(1,panel.canvas.winfo_height())/panel.image.height))
            panel.canvas.xview_moveto(0);panel.canvas.yview_moveto(0);panel.refresh()

    def view_state(self):
        return {n:dict(zoom=self.zooms[n],x=p.canvas.xview()[0],y=p.canvas.yview()[0]) for n,p in self.panels.items()}

    def restore(self,state):
        for n,v in state.items():self.zooms[n]=v['zoom']
        self.layout()
        for n,v in state.items():
            self.panels[n].canvas.xview_moveto(v['x']);self.panels[n].canvas.yview_moveto(v['y']);self.panels[n].refresh()
