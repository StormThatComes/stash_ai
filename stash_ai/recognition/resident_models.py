"""Run-owned lazy model stages and an optional byte-bounded CPU weight pool."""
from .profiling import pipeline_timing
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
        self.limit=limit;self.parked=OrderedDict();self.used=0

    def take(self,key,loader,device):
        if key not in self.parked:return loader()
        model,modules,size=self.parked.pop(key);self.used-=size
        try:
            for module in modules:module.to(device)
            return model
        except BaseException:
            release(model);raise

    def put(self,key,model,device):
        # Only explicit adapters with no pending readbacks or device-specific workers.
        modules=getattr(type(model),'resident_modules',None)
        if not self.limit or not str(device).startswith('cuda') or modules is None:
            release(model);return
        modules=modules(model)
        size=sum(t.numel()*t.element_size() for m in modules for t in (*m.parameters(),*m.buffers()))
        if not size or size>self.limit or key in self.parked:release(model);return
        while self.used+size>self.limit:
            _,(old,_,n)=self.parked.popitem(last=False);self.used-=n;release(old)
        try:
            for module in modules:module.to('cpu')
        except BaseException:
            release(model);raise
        self.parked[key]=(model,modules,size);self.used+=size

    def close(self):
        from stash_ai.runtime.cleanup import cleanup
        models=list(self.parked.values());self.parked.clear();self.used=0
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
