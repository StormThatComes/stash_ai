"""Bounded sequential face/body phases with explicit input ownership."""
from contextlib import ExitStack
from stash_ai.runtime.cleanup import cleanup


class Borrowed:
    def __init__(self,value):self.value=value
    def __getattr__(self,name):return getattr(self.value,name)
    def __enter__(self):return self.value
    def __exit__(self,*error):pass


class Batches:
    def __init__(self,ordered,size,release_faces,release_all):
        self.ordered,self.size=ordered,size
        self.release_faces,self.release_all=release_faces,release_all
        self.inputs={};self.results={};self.stack=None
    def events(self):
        if self.size<=1:
            for position,identifier in enumerate(self.ordered):yield 'combined',position,identifier
            return
        for start in range(0,len(self.ordered),self.size):
            self.inputs={};self.results={}
            with ExitStack() as stack:
                self.stack=stack
                try:
                    stop=min(start+self.size,len(self.ordered))
                    for position in range(start,stop):yield 'face',position,self.ordered[position]
                    self.release_faces()
                    for position in range(start,stop):
                        if position in self.results:yield 'body',position,self.ordered[position]
                finally:
                    cleanup([('release batched inference models',self.release_all)])
                    self.inputs.clear();self.results.clear();self.stack=None
    def take(self,phase,position,identifier,inputs):
        if phase=='combined':return inputs.take(identifier)
        if phase=='face':self.inputs[position]=self.stack.enter_context(inputs.take(identifier))
        return Borrowed(self.inputs[position])
