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
from stash_ai.runtime.cleanup import cleanup, closing
from .media import safe_url
from .models import checked_detection
from .visibility import cached as visible,required as needs_visible
from .portrait_admission import extra_measurements,pose_reason
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
        from .sprite_cache import current
        self.sprite_cache = current()
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
        from .sprite_cache import use as use_sprites
        with use(self.cancel), use_sprites(self.sprite_cache):return fn(*args)

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
                for data in getattr(source,'cached_bytes',getattr(source,'cache',{}).values()):
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
            cost=sum(len(data) for data in getattr(source,'cached_bytes',getattr(source,'cache',{}).values()))
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
    def __init__(self,model,executor,cpu=None,cpu_limit=0):
        self.cpu,self.cpu_limit=cpu,cpu_limit
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
            with closing(owned,name='release portrait model pixels'):return self.model.measure(owned,face)
        try:future=self.executor.submit(invoke)
        except BaseException:
            owned.close();raise
        future.add_done_callback(lambda f:owned.close() if f.cancelled() else None)
        try:return future.result()
        except BaseException:
            future.cancel();raise

    def measure_many(self,items):
        from .visibility import LazyVisibility,FaceVisibility,prepare,VisibilityGeometryError
        self.prepare_assets()
        if len(items)>8:raise ValueError('Portrait model microbatch exceeds eight faces.')
        if isinstance(self.model,LazyVisibility) and self.model.factory is FaceVisibility:
            prepared=[]
            size=sum(image.width*image.height*3*8 + 1048576 for image,_ in items)
            if self.cpu is not None and len(items)>1 and size<=self.cpu_limit:
                from .portrait_cpu import visibility_input
                futures=[]
                try:
                    for image,face in items:
                        owned=image.copy()
                        try:future=self.cpu.submit(visibility_input,owned,face)
                        except BaseException:
                            owned.close();raise
                        future.add_done_callback(lambda f,owned=owned:owned.close() if f.cancelled() else None)
                        futures.append(future)
                    prepared=[future.result() for future in futures]
                finally:
                    for future in futures:future.cancel()
            else:
                for image,face in items:
                    try:prepared.append(prepare(image,face))
                    except VisibilityGeometryError as error:prepared.append(error)
            return self.executor.submit(self.model.measure_prepared_many,prepared).result()
        owned=[(image.copy(),face) for image,face in items]
        def invoke():
            try:
                many=getattr(self.model,'measure_many',None)
                if callable(many):return many(owned)
                values=[]
                for image,face in owned:
                    try:values.append(self.model.measure(image,face))
                    except Exception as error:values.append(error)
                return values
            finally:cleanup([image.close for image,_ in owned])
        try:future=self.executor.submit(invoke)
        except BaseException:
            cleanup([image.close for image,_ in owned]);raise
        future.add_done_callback(lambda f:cleanup([image.close for image,_ in owned]) if f.cancelled() else None)
        try:return future.result()
        except BaseException:
            future.cancel();raise


class Pipeline:
    def __init__(self,database,client,config,scope,targets,source,visibility,expression,*,enabled=True):
        self.db,self.client,self.config,self.scope,self.targets=database,client,config,scope,targets
        self.source_factory=source
        self.enabled=(enabled and config.data['execution'].get('prefetch_portraits',True)
                      and config.data['execution'].get('portrait_cpu_threads',0)!=1
                      and config.data['portraits']['enabled'] and config.data['portraits']['kind']!='person' and config.data['portraits']['source']=='vtt'
                      and callable(getattr(client,'new_media_reader',None)))
        self.original_visibility,self.original_expression=visibility,expression
        self.visibility,self.expression=visibility,expression
        self.cancel=Cancellation()
        self.pending=None;self.upcoming=[];self.upcoming_target=None;self.prepared={};self.initial_reads={};self.index=-1
        self.preflight_future=None;self.prime_future=None;self.completed=self.pending_total=0
        self.models=self.preparer=self.sources=self.cpu=None
        self.identity_scores={}

    def __enter__(self):
        self.context=_active.set(self if self.enabled else None)
        try:
            if self.enabled:
                limit=min(268435456,self.config.data['execution']['rebuild_memory_bytes']//16)
                self.descriptor_limit=limit//8
                self.models=ThreadPoolExecutor(max_workers=1,thread_name_prefix='portrait-models')
                self.preparer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='portrait-quality')
                from .portrait_threads import preparation_workers
                self.cpu_workers=preparation_workers(self.config)
                if self.cpu_workers:
                    self.cpu=ThreadPoolExecutor(max_workers=self.cpu_workers,thread_name_prefix='portrait-cpu')
                self.visibility=ModelProxy(self.original_visibility,self.models,self.cpu,limit//8)
                self.expression=ModelProxy(self.original_expression,self.models)
                self.sources=SourceOwner(self.client,self.config,self.source_factory,limit,self.cancel)
        except BaseException:
            _active.reset(self.context)
            cleanup([('close partial portrait sources',lambda:self.sources.close() if self.sources is not None else None),
                     ('close partial portrait preparation',lambda:self.preparer.shutdown(wait=True,cancel_futures=True) if self.preparer is not None else None),
                     ('close partial portrait CPU',lambda:self.cpu.shutdown(wait=True,cancel_futures=True) if self.cpu is not None else None),
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
        def pairs():
            for item in sorted(target['scans'],key=lambda s:(s['scene'],s['scan'])):
                rows=db.execute('SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,o.quality,'
                    'o.dimensions,o.embedding,s.input_sha FROM observations o JOIN scans s ON s.id=o.scan '
                    'JOIN members m ON m.observation=o.id WHERE m.person=? AND o.scan=? AND s.scope=? '
                    "AND s.scene=? AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL ORDER BY o.cue,o.id",
                    (target['person'],item['scan'],self.scope,item['scene'],person['namespace']))
                for row,candidate in candidate_observations(db,rows,item['scan'],matching_policy(self.config)):
                    yield item['scene'],row,candidate
        return self._build(target,groups,context,pairs())

    def _build(self,target,groups,context,pairs):
        from .portrait_availability import observation_identity,crop_row,measurement_basis
        from .database import encoded
        db=self.db.connection;policy=self.config.data['portraits'];result=[];size=0
        required={name for name,needed in [('visibility',needs_visible(policy)),('expression',needs_expression(policy))] if needed}
        required.update(extra_measurements(self.config))
        cached={'visibility':visible,'expression':expressive}
        for visited,(scene,row,candidate) in enumerate(pairs):
            if visited>=128:break
            if self.outcome(row,context) is not None:continue
            quality=json.loads(row['quality']);bases=quality.setdefault('measurement_basis',{})
            if pose_reason(quality,self.config):continue
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
            owned=dict(row)
            cost=2048+sum(len(v)*4 if isinstance(v,str) else len(v) if isinstance(v,bytes) else 64 for v in owned.values())
            if size+cost>self.descriptor_limit:break
            size+=cost
            result.append((scene,owned,quality,missing,approximate,identity,context))
        # Read fallback JPEGs only in bounded ordered sets, never in a worker.
        ids=[]
        for row in db.execute('SELECT o.id,LENGTH(o.crop) AS bytes FROM json_each(?) chosen '
            'JOIN observations o ON o.id=chosen.value ORDER BY chosen.key',
            (encoded([item[1]['id'] for item in result]),)):
            cost=row['bytes'] or 0
            if size+cost>self.descriptor_limit:break
            size+=cost;ids.append(row['id'])
        allowed=set(ids)
        result=[item for item in result if item[1]['id'] in allowed]
        crops={row['id']:row['crop'] for row in db.execute('SELECT id,crop FROM observations WHERE id IN '
            '(SELECT value FROM json_each(?))',(encoded(ids),))}
        for item in result:item[1]['crop']=crops[item[1]['id']]
        return result

    def begin_ranking(self,target,groups,context):
        self.ranking=target,groups,context
        self.identity_scores={}

    def rank_batch(self,rows):
        if not getattr(self,'ranking',None) or self.cpu_workers<=1:return
        from .portrait_cpu import identity_batch
        target,groups,context=self.ranking
        self.identity_scores={}
        futures=[]
        try:
            for start in range(0,len(rows),16):
                batch=rows[start:start+16]
                payload=[{k:r[k] for k in ('dimensions','embedding')} for r in batch]
                futures.append((batch,self.cpu.submit(identity_batch,(payload,groups,target['person'],self.config,self.cancel))))
            from .progress import stage
            from stash_ai.runtime.console import current_report, status
            completed=0
            with stage('Compare portrait identity batches'):
                for batch,future in futures:
                    for row,value in zip(batch,future.result()):
                        if value is not None:
                            self.identity_scores[(row['id'],row['dimensions'],row['embedding'])]=value
                    completed+=len(batch)
                    status(current_report(),portrait_cpu=f'{completed}/{len(rows)} observations')
        finally:
            from stash_ai.runtime.console import current_report, status
            status(current_report(),portrait_cpu=None)
            for _,future in futures:future.cancel()

    def identity_outcome(self,row,groups,target,context):
        if not getattr(self,'ranking',None):return None
        selected,bank,stamp=self.ranking
        if groups is not bank or selected['person']!=target['person'] or stamp!=context:return None
        return self.identity_scores.get((row['id'],row['dimensions'],row['embedding']))

    def ranking_finished(self):
        if self.config.data['execution'].get('portrait_cpu_threads',0) in (1,2):return
        if not getattr(self,'ranking',None):return
        target=self.ranking[0]
        if self.db.connection.execute('SELECT 1 FROM portraits WHERE scope=? AND person=? '
            'AND image IS NOT NULL AND json_valid(metadata) LIMIT 1',(self.scope,target['person'])).fetchone():return
        self.promote()

    def prepare_batch(self,scene,pairs):
        if not getattr(self,'ranking',None):return
        target,groups,context=self.ranking
        try:
            descriptors=self._build(target,groups,context,((scene,row,candidate) for row,candidate in pairs))
            if len(descriptors)<2:return
            if needs_visible(self.config.data['portraits']) or extra_measurements(self.config):self.visibility.prepare_assets()
            if needs_expression(self.config.data['portraits']):self.expression.prepare_assets()
            self.completed=0
            future=self.preparer.submit(self._prepare,descriptors)
            from .progress import work
            with work('Prepare VTT quality batches',len(descriptors),unit='observation') as progress:
                while True:
                    try:
                        self.prepared.update(future.result(timeout=.1))
                        progress.set_completed(self.completed);break
                    except TimeoutError:
                        if future.done():raise
                        progress.set_completed(self.completed)
        except Exception:
            pass  # Scalar demand evaluation retains its original error boundary.

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
            if needs_visible(self.config.data['portraits']) or extra_measurements(self.config):self.visibility.prepare_assets()
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
        from .portrait_availability import retained_image,measurement_basis
        from .diagnostics import message as safe_error
        result={};group=[];pixels=0
        maximum=self.config.data['execution'].get('portrait_faceocc_batch_size',4)
        allowance=self.descriptor_limit
        def finish():
            if not group:return
            try:
                selected=[entry for entry in group if entry['image'] is not None and 'visibility' in entry['requested']]
                if selected:
                    values=self.visibility.measure_many([(e['image'],e['face']) for e in selected])
                    for entry,value in zip(selected,values):
                        if isinstance(value,Exception):entry['errors']['visibility']=safe_error(value,getattr(self.client,'api_key',''))[:500]
                        else:entry['quality'].update(value)
                for entry in group:
                    if entry['image'] is not None and 'expression' in entry['requested']:
                        try:entry['quality'].update(self.expression.measure(entry['image'],entry['face']))
                        except Exception as error:entry['errors']['expression']=safe_error(error,getattr(self.client,'api_key',''))[:500]
                    checks={'visibility':visible,'expression':expressive}
                    for name in entry['requested']:
                        if entry['image'] is not None and name not in entry['errors'] and checks[name](entry['quality']):
                            entry['quality']['measurement_basis'][name]=measurement_basis(entry['row'],entry['source'])
                    row=entry['row']
                    result[row['id']]=dict(signature=signature(row),quality=entry['quality'],errors=entry['errors'],
                        warning=entry['warning'],identity=entry['identity'],context=entry['context'])
                    self.completed+=1
            finally:
                cleanup([e['image'].close for e in group if e['image'] is not None])
                group.clear()
        with use(self.cancel):
            try:
                for scene,row,quality,missing,approximate,identity,context in descriptors:
                    self.cancel.check()
                    rect=json.loads(row['rect']);estimated=rect[2]*rect[3]*3
                    if group and (len(group)>=maximum or pixels+estimated>allowance):finish();pixels=0
                    source='vtt';warning=None;face=None
                    try:
                        face=checked_detection(json.loads(row['geometry']))
                        image=self.thumbnail(scene,row)
                    except Capacity:
                        self.completed+=1;continue
                    except Exception:
                        warning='vtt_source_unavailable_or_changed';source='recognition_crop'
                        try:image,face=retained_image(row)
                        except Exception:image=None
                    size=image.width*image.height*3 if image is not None else 0
                    if group and pixels+size>allowance:finish();pixels=0
                    pixels+=size
                    group.append(dict(row=row,quality=quality,requested=missing|(approximate if source=='vtt' else set()),
                        identity=identity,context=context,image=image,face=face,source=source,warning=warning,errors={}))
                finish()
            finally:
                cleanup([e['image'].close for e in group if e['image'] is not None])
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
                if pose_reason(quality,self.config):continue
                if (((needs_visible(policy) or extra_measurements(self.config)) and not visible(quality)) or (needs_expression(policy) and not expressive(quality))
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
        from stash_ai.runtime.shutdown import call
        _active.reset(self.context)
        if self.enabled:
            cleanup([('cancel portrait preparation',self.cancel.cancel),
                     ('join portrait quality preparation',lambda:self.preparer.shutdown(wait=True,cancel_futures=True)),
                     ('close portrait source stage',self.sources.close),
                     ('join portrait CPU preparation',lambda:call(lambda:self.cpu.shutdown(wait=True,cancel_futures=True),self.cpu) if self.cpu is not None else None),
                     ('join portrait model owner',lambda:self.models.shutdown(wait=True,cancel_futures=True))])
        self.identity_scores.clear()
        self.prepared.clear();self.initial_reads.clear();self.upcoming.clear();self.pending=None
        self.preflight_future=None


def current():return _active.get()
