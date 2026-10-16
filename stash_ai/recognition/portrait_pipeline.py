"""Bounded read/preparation lookahead; SQLite and publication stay on the caller."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import ContextVar
from io import BytesIO
import json
import threading
from types import SimpleNamespace

from stash_ai.inference.image_io import open_image
from stash_ai.runtime.cancellation import Cancellation,use
from stash_ai.runtime.cleanup import cleanup
from .media import safe_url
from .models import checked_detection
from .visibility import cached as visible,required as needs_visible
from .expression import cached as expressive,required as needs_expression

_active=ContextVar('recognition_portrait_pipeline',default=None)
MISSING=object()


class Capacity(ValueError):
    pass


def signature(row):
    return tuple(row[name] for name in ('id','scan','cue','seconds','source','rect','geometry','quality',
                                        'dimensions','embedding','input_sha'))


class LimitedResponse:
    def __init__(self,response,limit):self.response,self.limit=response,limit
    def __getattr__(self,name):return getattr(self.response,name)
    def __enter__(self):self.response.__enter__();return self
    def __exit__(self,*error):return self.response.__exit__(*error)
    def iter_content(self,size):
        used=0
        for block in self.response.iter_content(size):
            used+=len(block)
            if used>self.limit:raise Capacity('Portrait preparation source exceeds its queue allowance.')
            yield block


class LimitedReader:
    def __init__(self,reader,limit):
        self.reader,self.limit,self.reads=reader,limit,0
        self.media=SimpleNamespace(get=self.get)
    def __getattr__(self,name):return getattr(self.reader,name)
    def get(self,*args,**kwargs):
        limit=min(self.limit,524288 if self.reads==0 else 16777216)
        self.reads+=1
        return LimitedResponse(self.reader.media.get(*args,**kwargs),limit)


class SourceOwner:
    def __init__(self,client,config,factory,limit,cancel):
        self.client,self.config,self.factory,self.limit,self.cancel=client,config,factory,limit,cancel
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='portrait-source')
        self.reader=None
        self.lookahead=[]
        self.sources=OrderedDict();self.scenes=OrderedDict();self.images=OrderedDict()
        self.capacity=OrderedDict();self.image_bytes=0

    def _reader(self):
        if self.reader is None:self.reader=self.client.new_media_reader()
        return self.reader

    def _call(self,fn,*args):
        with use(self.cancel):return fn(*args)

    def submit(self,fn,*args):return self.executor.submit(self._call,fn,*args)

    def preflight(self,targets):
        reader=self._reader();result={}
        batch=getattr(reader,'performers',None)
        if not callable(batch):return result
        rows=batch([t['performer'] for t in targets])
        scenes=getattr(reader,'scenes',None)
        if callable(scenes):
            identifiers=list(dict.fromkeys(t['scans'][0]['scene'] for t in targets if t['scans']))
            for key,value in scenes(identifiers).items():
                if isinstance(value,dict):self.scenes[key]=value
            while len(self.scenes)>32:self.scenes.popitem(last=False)
        from .portraits import image_digest
        for target in targets:
            live=rows.get(target['performer'])
            if isinstance(live,dict):
                try:
                    from .stash import checked_id
                    checked_id(live,target['performer'])
                    result[target['person']]=(live,image_digest(reader,live,target['performer'],self.config))
                except Exception:pass
        return result

    def _scene(self,scene):
        reader=self._reader()
        value=self.scenes.pop(scene,None)
        if value is None:
            batch=getattr(reader,'scenes',None)
            value=batch([scene]).get(scene) if callable(batch) else reader.scene(scene)
            if isinstance(value,Exception):raise value
            if not isinstance(value,dict):raise ValueError('Portrait scene is unavailable.')
        self.scenes[scene]=value
        while len(self.scenes)>32:self.scenes.popitem(last=False)
        return value

    def thumbnail(self,scene,row):
        self.cancel.check()
        key=signature(row)
        cached=self.images.pop(key,None)
        if cached is not None:
            self.image_bytes-=cached.width*cached.height*3
            return cached
        if (scene,row['input_sha']) in self.capacity:raise Capacity('Use serial preparation for this source.')
        source=self.sources.pop(scene,None)
        try:
            if source is None:
                if len(self.sources)>=2:
                    _,old=self.sources.popitem(last=False);old.close()
                source=self.factory(LimitedReader(self._reader(),max(1,self.limit//16)),self._scene(scene),self.config).__enter__()
                if hasattr(source,'portrait_sheet_bytes'):
                    source.portrait_sheet_bytes=min(source.portrait_sheet_bytes,self.limit//8)
                for data in getattr(source,'cache',{}).values():
                    with open_image(BytesIO(data)) as raw:
                        if raw.width*raw.height*4>self.limit//4:
                            raise Capacity('Decoded portrait sheet exceeds preparation allowance.')
            if source.fingerprint!=row['input_sha']:raise ValueError('VTT source changed after recognition.')
            cue=source.cues[row['cue']]
            if (cue.start!=row['seconds'] or safe_url(cue.image)!=safe_url(row['source'])
                    or list(cue.rect)!=json.loads(row['rect'])):
                raise ValueError('VTT cue does not match its recognition observation.')
            image=source.thumbnail(row['cue'])
            if image.width*image.height*3>self.limit//4:
                image.close();raise Capacity('Portrait thumbnail exceeds preparation allowance.')
            cost=sum(len(data) for data in getattr(source,'cache',{}).values())
            cost+=len(getattr(source,'cues',()))*512
            sheet=getattr(source,'_portrait_sheet',None)
            if sheet is not None and sheet[2] is not None:cost+=sheet[2].width*sheet[2].height*3
            if cost>self.limit//4:
                source.close();return image
            self.sources[scene]=source
            while len(self.sources)>2:
                _,old=self.sources.popitem(last=False);old.close()
            return image
        except BaseException as error:
            if isinstance(error,Capacity):
                self.capacity[scene,row['input_sha']]=None
                while len(self.capacity)>128:self.capacity.popitem(last=False)
            if source is not None:cleanup([('close failed portrait source',source.close)])
            raise

    def prime(self,descriptors):
        self.lookahead=list(descriptors[:128])
        self.prefetch(self.lookahead)

    def after(self,observation):
        for index,(_,row,*_) in enumerate(self.lookahead):
            if row['id']==observation:
                self.lookahead=self.lookahead[index+1:]
                self.prefetch(self.lookahead)
                break

    def prefetch(self,descriptors):
        for scene,row,*_ in descriptors[:2]:
            if signature(row) in self.images:continue
            try:image=self.thumbnail(scene,row)
            except Exception:continue
            size=image.width*image.height*3
            while self.images and (self.image_bytes+size>self.limit//4 or len(self.images)>=8):
                _,old=self.images.popitem(last=False);self.image_bytes-=old.width*old.height*3;old.close()
            if size>self.limit//4:image.close();continue
            key=signature(row)
            old=self.images.pop(key,None)
            if old is not None:self.image_bytes-=old.width*old.height*3;old.close()
            self.images[key]=image;self.image_bytes+=size

    def close(self):
        def release():
            actions=[('close prepared thumbnail',image.close) for image in self.images.values()]
            actions += [('close shared portrait scene',source.close) for source in self.sources.values()]
            if self.reader is not None:actions.append(('close portrait read client',self.reader.close))
            cleanup(actions)
            self.images.clear();self.sources.clear();self.scenes.clear()
        cleanup([('release portrait source owner',lambda:self.executor.submit(release).result()),
                 ('join portrait source owner',lambda:self.executor.shutdown(wait=True,cancel_futures=True))])


class ModelProxy:
    def __init__(self,model,executor):
        self.model,self.executor=model,executor
        self.owner=threading.get_ident();self.assets_ready=False

    def prepare_assets(self):
        if self.assets_ready:return
        if threading.get_ident()!=self.owner:
            raise RuntimeError('Portrait assets must be prepared by the coordinator.')
        from .visibility import FaceVisibility
        from .expression import FaceExpression
        factory=getattr(self.model,'factory',None)
        keys=('faceocc',) if factory is FaceVisibility else ('expression_mesh','expression_blendshapes') if factory is FaceExpression else ()
        try:
            for key in keys:self.model.paths[key]
        except Exception:
            pass  # Lazy model ownership re-raises its original cached asset error.
        self.assets_ready=True

    def measure(self,image,face):
        self.prepare_assets()
        owned=image.copy()
        def invoke():
            with owned:return self.model.measure(owned,face)
        try:future=self.executor.submit(invoke)
        except BaseException:
            owned.close();raise
        future.add_done_callback(lambda f:owned.close() if f.cancelled() else None)
        try:return future.result()
        except BaseException:
            future.cancel();raise


class Pipeline:
    def __init__(self,database,client,config,scope,targets,source,visibility,expression,*,enabled=True):
        self.db,self.client,self.config,self.scope,self.targets=database,client,config,scope,targets
        self.source_factory=source
        self.enabled=(enabled and config.data['execution'].get('prefetch_portraits',True)
                      and config.data['portraits']['enabled'] and config.data['portraits']['source']=='vtt'
                      and callable(getattr(client,'new_media_reader',None)))
        self.original_visibility,self.original_expression=visibility,expression
        self.visibility,self.expression=visibility,expression
        self.cancel=Cancellation()
        self.pending=None;self.upcoming=[];self.upcoming_target=None;self.prepared={};self.initial_reads={};self.index=-1
        self.preflight_future=None;self.prime_future=None;self.completed=self.pending_total=0
        self.models=self.preparer=self.sources=None

    def __enter__(self):
        self.context=_active.set(self if self.enabled else None)
        try:
            if self.enabled:
                limit=min(268435456,self.config.data['execution']['rebuild_memory_bytes']//16)
                self.descriptor_limit=limit//8
                self.models=ThreadPoolExecutor(max_workers=1,thread_name_prefix='portrait-models')
                self.preparer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='portrait-quality')
                self.visibility=ModelProxy(self.original_visibility,self.models)
                self.expression=ModelProxy(self.original_expression,self.models)
                self.sources=SourceOwner(self.client,self.config,self.source_factory,limit,self.cancel)
        except BaseException:
            _active.reset(self.context)
            cleanup([('close partial portrait sources',lambda:self.sources.close() if self.sources is not None else None),
                     ('close partial portrait preparation',lambda:self.preparer.shutdown(wait=True,cancel_futures=True) if self.preparer is not None else None),
                     ('close partial portrait models',lambda:self.models.shutdown(wait=True,cancel_futures=True) if self.models is not None else None)])
            raise
        return self

    def _descriptors(self,target):
        from .context import portrait_references
        from .clustering import _groups,matching_policy
        from .portrait_availability import observation_identity,candidate_observations,crop_row,measurement_basis
        db=self.db.connection;policy=self.config.data['portraits']
        if db.execute('SELECT 1 FROM portraits WHERE scope=? AND person=? AND image IS NOT NULL '
                      'AND json_valid(metadata) LIMIT 1',(self.scope,target['person'])).fetchone():
            return []  # Existing targets retain demand-driven ranking/retry without a history prepass.
        person=db.execute('SELECT namespace FROM people WHERE id=? AND scope=?',(target['person'],self.scope)).fetchone()
        groups,context=portrait_references(self.db,self.scope,person['namespace'],self.config,
            lambda:_groups(db,self.scope,person['namespace'],self.config.data['clustering']))
        result=[];size=visited=0
        required={name for name,needed in [('visibility',needs_visible(policy)),('expression',needs_expression(policy))] if needed}
        cached={'visibility':visible,'expression':expressive}
        for item in sorted(target['scans'],key=lambda s:(s['scene'],s['scan'])):
            rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,'
                'o.dimensions,o.embedding,s.input_sha FROM observations o JOIN scans s ON s.id=o.scan '
                'JOIN members m ON m.observation=o.id WHERE m.person=? AND o.scan=? AND s.scope=? '
                "AND s.scene=? AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL ORDER BY o.cue,o.id",
                (target['person'],item['scan'],self.scope,item['scene'],person['namespace']))
            for row,candidate in candidate_observations(db,rows,item['scan'],matching_policy(self.config)):
                visited+=1
                if visited>128:return result
                quality=json.loads(row['quality']);bases=quality.setdefault('measurement_basis',{})
                legacy=None
                for name in required:
                    if cached[name](quality) and name not in bases:
                        if legacy is None:
                            legacy=db.execute("SELECT 1 FROM portraits WHERE scope=? AND person=? AND json_valid(metadata) "
                                "AND json_extract(metadata,'$.observation')=? AND json_extract(metadata,'$.quality_source')='recognition_crop' LIMIT 1",
                                (self.scope,target['person'],row['id'])).fetchone()
                        bases[name]=measurement_basis(crop_row(db,row) if legacy else row,'recognition_crop' if legacy else 'vtt')
                missing={n for n in required if not cached[n](quality)}
                approximate={n for n in required if bases.get(n,{}).get('source_type')=='recognition_crop'}
                if not (missing or approximate):continue
                try:identity=observation_identity(self.db,row,groups,target,self.config,context,candidate=candidate)
                except (ValueError,TypeError):continue
                if identity[0]:continue
                owned=dict(crop_row(db,row))
                cost=2048+sum(len(v)*4 if isinstance(v,str) else len(v) if isinstance(v,bytes) else 64 for v in owned.values())
                if len(result)>=128 or size+cost>self.descriptor_limit:return result
                size+=cost
                result.append((item['scene'],owned,quality,missing,approximate,identity,context))
        return result

    def activate(self,index):
        self.index=index;self.prepared={}
        if not self.enabled:return
        if self.pending is not None:
            from .progress import work
            try:
                with work('Prepare next VTT target',self.pending_total,unit='observation') as progress:
                    while True:
                        try:
                            self.prepared=self.pending.result(timeout=.1)
                            progress.set_completed(self.completed);break
                        except TimeoutError:
                            if self.pending.done():raise
                            progress.set_completed(self.completed)
            except Exception:self.prepared={}
            self.pending=None
        if self.preflight_future is not None:
            try:self.initial_reads=self.preflight_future.result()
            except Exception:self.initial_reads={}
        future_targets=self.targets[index:index+3]
        self.preflight_future=self.sources.submit(self.sources.preflight,future_targets)
        self.upcoming=[];self.upcoming_target=None
        if index+1<len(self.targets):
            self.upcoming_target=self.targets[index+1]
            try:self.upcoming=self._descriptors(self.targets[index+1])
            except Exception:self.upcoming=[]
            if self.upcoming:self.sources.submit(self.sources.prime,self.upcoming)

    def initial(self,target):
        if not self.enabled:return MISSING
        if target['person'] not in self.initial_reads and self.preflight_future is not None:
            try:self.initial_reads.update(self.preflight_future.result())
            except Exception:return MISSING
        return self.initial_reads.get(target['person'],MISSING)

    def promote(self):
        if not (self.enabled and self.upcoming and self.pending is None):return
        try:
            target=self.upcoming_target
            preflight=self.initial(target)
            if preflight is MISSING or not self._managed(target,preflight):
                self.upcoming=[];return
            if needs_visible(self.config.data['portraits']):self.visibility.prepare_assets()
            if needs_expression(self.config.data['portraits']):self.expression.prepare_assets()
            self.sources.submit(self.sources.prime,self.upcoming)
            self.completed=0;self.pending_total=len(self.upcoming)
            self.pending=self.preparer.submit(self._prepare,self.upcoming)
            self.upcoming=[]
        except Exception:
            self.upcoming=[]  # An optional future target cannot fail current publication.

    def _managed(self,target,preflight):
        from .portraits import missing_image
        from .stash import owns,checked_token
        from .publication import operation_id
        row=self.db.connection.execute('SELECT token FROM people WHERE id=? AND scope=?',
            (target['person'],self.scope)).fetchone()
        if row is None:return False
        checked_token(self.db,target['person'],row['token'])
        if not owns(preflight[0],row['token']):return False
        previous=self.db.connection.execute('SELECT * FROM publications WHERE id=?',
            (operation_id(self.scope,'',target['person'],'image'),)).fetchone()
        if previous is None:return missing_image(preflight[0],self.client,target['performer'])
        if previous['status'] in ('pending','unknown'):return False
        if preflight[1] is None:return True
        expected=(json.loads(previous['payload']).get('sha256') if previous['status'] in ('confirmed','observed')
                  else json.loads(previous['precondition']).get('image_sha256') if previous['status']=='failed' else None)
        return preflight[1]==expected

    def _prepare(self,descriptors):
        from .portrait_availability import retained_image,measure_available,measurement_basis
        result={};quiet=SimpleNamespace(api_key=getattr(self.client,'api_key',''),report=None)
        with use(self.cancel):
            for scene,row,quality,missing,approximate,identity,context in descriptors:
                self.cancel.check();source='vtt';warning=None
                try:
                    face=checked_detection(json.loads(row['geometry']))
                    image=self.thumbnail(scene,row)
                except Capacity:
                    self.completed+=1;continue
                except Exception:
                    warning='vtt_source_unavailable_or_changed';source='recognition_crop'
                    try:image,face=retained_image(row)
                    except Exception:image=None
                errors={}
                if image is not None:
                    force=approximate if source=='vtt' else set()
                    with image:
                        errors=measure_available(quiet,image,face,quality,self.config.data['portraits'],
                                                 self.visibility,self.expression,force=force)
                    checks={'visibility':visible,'expression':expressive}
                    for name in missing|force:
                        if name not in errors and checks[name](quality):
                            quality['measurement_basis'][name]=measurement_basis(row,source)
                result[row['id']]=dict(signature=signature(row),quality=quality,errors=errors,warning=warning,
                                       identity=identity,context=context)
                self.completed+=1
        return result

    def outcome(self,row,context):
        found=self.prepared.get(row['id'])
        return found if found is not None and found['signature']==signature(row) and found['context']==context else None

    def prime(self,scene,rows):
        if self.prime_future is not None and not self.prime_future.done():return
        needed=[];size=0
        policy=self.config.data['portraits']
        for row in rows[:128]:
            try:
                quality=json.loads(row['quality'])
                if ((needs_visible(policy) and not visible(quality)) or (needs_expression(policy) and not expressive(quality))
                        or any(v.get('source_type')=='recognition_crop' for v in quality.get('measurement_basis',{}).values())):
                    cost=2048+sum(len(v)*4 if isinstance(v,str) else len(v) if isinstance(v,bytes) else 64 for v in dict(row).values())
                    if size+cost>self.descriptor_limit:break
                    needed.append((scene,dict(row)));size+=cost
            except (ValueError,TypeError,AttributeError):pass
        if needed:self.prime_future=self.sources.submit(self.sources.prime,needed)

    def thumbnail(self,scene,row):
        future=self.sources.submit(self.sources.thumbnail,scene,dict(row))
        try:
            image=future.result()
            self.sources.submit(self.sources.after,row['id'])
            return image
        except BaseException:
            def release(future):
                if not future.cancelled() and future.exception() is None:future.result().close()
            future.add_done_callback(release)
            raise

    def __exit__(self,*error):
        _active.reset(self.context)
        if self.enabled:
            cleanup([('cancel portrait preparation',self.cancel.cancel),
                     ('join portrait quality preparation',lambda:self.preparer.shutdown(wait=True,cancel_futures=True)),
                     ('close portrait source stage',self.sources.close),
                     ('join portrait model owner',lambda:self.models.shutdown(wait=True,cancel_futures=True))])
        self.prepared.clear();self.initial_reads.clear();self.upcoming.clear();self.pending=None
        self.preflight_future=None


def current():return _active.get()
