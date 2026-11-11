"""Run-owned lazy model stages and an optional byte-bounded CPU weight pool."""
from .profiling import pipeline_timing, pipeline_count
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

_active=ContextVar('recognition_models',default=None)


def current():return _active.get()


def release(model):
    close=getattr(model,'close',None)
    if callable(close):close()


class Pool:
    def __init__(self,limit):
        from .retained_memory import current
        self.budget = current()
        self.limit=limit;self.parked=OrderedDict();self.used=0

    def take(self,key,loader,device):
        if key not in self.parked:
            pipeline_count('resident_model_loads')
            with pipeline_timing('resident_model_load'):return loader()
        model,modules,size=self.parked.pop(key);self.used-=size
        try:
            with pipeline_timing('resident_model_restore'):
                for module in modules:module.to(device)
            pipeline_count('resident_model_restores')
            return model
        except BaseException:
            release(model);raise
        finally:
            if self.budget is not None:self.budget.replace(self,self.used)

    def put(self,key,model,device):
        # Only explicit adapters with no pending readbacks or device-specific workers.
        modules=getattr(type(model),'resident_modules',None)
        if not self.limit or not str(device).startswith('cuda') or modules is None:
            release(model);return
        modules=modules(model)
        size=sum(t.numel()*t.element_size() for m in modules for t in (*m.parameters(),*m.buffers()))
        if not size or size>self.limit or key in self.parked:release(model);return
        while self.parked and (self.used+size>self.limit or
                self.budget is not None and not self.budget.replace(self,self.used+size)):
            _,(old,_,n)=self.parked.popitem(last=False);self.used-=n;release(old)
        if self.budget is not None and not self.budget.replace(self,self.used+size):
            self.budget.replace(self,self.used)
            release(model);return
        try:
            prepare=getattr(model,'prepare_residency',None)
            if callable(prepare):prepare()
            with pipeline_timing('resident_model_park'):
                for module in modules:module.to('cpu')
        except BaseException:
            if self.budget is not None:self.budget.replace(self,self.used)
            release(model);raise
        self.parked[key]=(model,modules,size);self.used+=size

    def close(self):
        from stash_ai.runtime.cleanup import cleanup
        models=list(self.parked.values());self.parked.clear();self.used=0
        if self.budget is not None:self.budget.replace(self,0)
        cleanup([('close parked model',lambda m=m:release(m)) for m,_,_ in models])


@contextmanager
def scope(config):
    ex=config.data['execution']
    pool=Pool(ex.get('cpu_residency_bytes',0) if ex['gpu_model_policy']=='sequential' else 0)
    token=_active.set(pool)
    try:yield pool
    finally:
        try:pool.close()
        finally:_active.reset(token)


class Stage:
    def __init__(self,key,loader,device):
        self.key,self.loader,self.device=key,loader,device
        self.pool=current();self.model=None;self.failed=False

    def invoke(self,name,*args,**kwargs):
        if self.model is None:
            stage = {'person_detector': 'person_detector_model_acquire',
                     'pose_refiner': 'pose_model_acquire',
                     'person_encoder': 'body_model_acquire'}.get(self.key, 'other_model_acquire')
            with pipeline_timing(stage):
                self.model=self.pool.take(self.key,self.loader,self.device)
        try:
            if name=='predict_many' and not hasattr(self.model,name):
                return [self.model.predict(*row) for row in args[0]]
            return getattr(self.model,name)(*args,**kwargs)
        except BaseException:
            self.failed=True;raise

    def predict(self,*args,**kwargs):return self.invoke('predict',*args,**kwargs)
    def predict_many(self,*args):return self.invoke('predict_many',*args)
    def predict_frames(self,frames):return self.invoke('predict_frames',frames)
    def frame(self,image):return self.invoke('frame',image)

    def close(self):
        model,self.model=self.model,None
        if model is None:return
        if self.failed:release(model)
        else:self.pool.put(self.key,model,self.device)


class Encoder(Stage):
    def __init__(self,config,loader,device):
        from .inference_contracts import body_encoder
        super().__init__('person_encoder',loader,device)
        self.contract=body_encoder(config);self.max_batch_size=config.data['sapiensid']['execution']['batch_size']

    def embed_prepared(self,prepared):
        if not 1<=len(prepared)<=self.max_batch_size:raise ValueError('Invalid SapiensID batch.')
        from .inference_cache import many,using
        def compute(values):
            # The outer lookup owns the cache; the actual encoder still owns OOM recovery.
            with using(None):return self.invoke('embed_prepared',values)
        return many('body_embedding',[self.contract,str(self.device)],prepared,compute)

    def embed_person_batch(self,persons):
        from .sapiensid_v1 import prepare_person_input
        return self.embed_prepared([prepare_person_input(image,points) for image,points in persons])
