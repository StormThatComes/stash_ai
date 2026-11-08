"""Owned CPU preparation with cross-cue batches and complete-cue publication."""
from collections import deque
from functools import cached_property
from concurrent.futures import Future
from io import BytesIO
from .person_signal import save, contract
from .scan_pipeline import CpuStage


class PreparedCollector:
    def __init__(self, config, model, writer=None):
        self.config,self.model=config,model
        self.writer=writer
        settings=config.data['execution']
        self.stage=CpuStage(settings['scan_cpu_workers'],settings['scan_queue_bytes']//2)
        self.step=max(1,min(config.data['sapiensid']['execution']['batch_size'],
                            self.stage.limit//(2*(3*384*384*4+384*384+config.data['storage']['max_crop_bytes']+4096))))
        self.cues=deque();self.jobs=[];self.db=None

    @cached_property
    def person_contract(self):
        return contract(self.config)

    def prepare(self,image,value,ready=None):
        bounds,points,*mask=value
        from .sapiensid_v1 import prepare_person_input
        from .person_geometry import encoder_crop
        with encoder_crop(image,bounds,*mask) as input_crop:
            prepared=prepare_person_input(input_crop,points)
        if ready is not None:
            ready.set_result(prepared)
        encoded = None
        omitted = False
        if self.config.data['sapiensid']['storage']['retain_person_crops'] and self.config.data['storage']['crop_policy']=='all':
            with image.crop(bounds) as crop, BytesIO() as output:
                crop.save(output, format='PNG')
                if output.tell() <= self.config.data['storage']['max_crop_bytes']:
                    encoded = output.getvalue()
                else:
                    omitted = True
        return prepared, encoded, omitted

    def _prepare_job(self, image, job):
        value, ready = job
        try:
            prepared, crop, omitted = self.prepare(image, value, ready)
            if not ready.done():
                ready.set_result(prepared)
            return crop, omitted
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            raise

    def _submit(self, cue, record, image, value):
        extra = 3*384*384*4 + 384*384 + self.config.data['storage']['max_crop_bytes']
        while self.jobs and not self.stage.can_submit(image, extra=extra):
            self.infer()
        if not self.stage.can_submit(image, extra=extra):
            self.commit(wait=True)
            # An unsealed dense cue may own completed encodings; release their
            # reservations before admitting another person in the same cue.
            for pending in self.cues:
                if self.stage.can_submit(image, extra=extra):
                    break
                self._take_crops(pending)
        ready = Future()
        crop = self.stage.submit(self._prepare_job, image, (value, ready), extra=extra)
        def propagate_failure(done):
            # Cancellation can occur before CpuStage enters the pixel callback.
            if not ready.done():
                if done.cancelled():
                    ready.cancel()
                else:
                    error = done.exception()
                    if error is not None:
                        ready.set_exception(error)
        crop.add_done_callback(propagate_failure)
        cue.setdefault('crops', []).append((record, crop))
        cue['remaining'] += 1
        self.jobs.append((cue, record, ready))
        if len(self.jobs) >= 2*self.step:
            self.infer()

    def collect(self,db,image,rows,faces,*,fresh=False):
        if self.db is not None and self.db is not db:raise RuntimeError('Person pipeline database owner changed.')
        self.db=db;cue=dict(rows=[],remaining=0,sealed=False);self.cues.append(cue)
        for row,face in zip(rows,faces):
            if row['rejected'] is not None:continue
            if not fresh and db.execute('SELECT 1 FROM person_evidence WHERE observation=? AND contract=?',(row['id'],self.person_contract)).fetchone():continue
            metadata=self.model.observation_quality(image,face)
            record=[row['id'],metadata,None,None];cue['rows'].append(record)
            if metadata['identity_rejection'] is None:
                value=self.model.person_input(image,face)
                self._submit(cue, record, image, value)
        cue['sealed']=True
        if len(self.cues)>=2*self.step:self.infer()
        self.commit(wait=len(self.cues)>=2*self.step)

    def collect_items(self,db,image,items):
        if self.db is not None and self.db is not db:raise RuntimeError('Person pipeline database owner changed.')
        self.db=db;cue=dict(rows=[],remaining=0,sealed=False);self.cues.append(cue)
        for record,value in items:
            cue['rows'].append(record)
            if value is not None:
                self._submit(cue, record, image, value)
        cue['sealed']=True
        if len(self.cues)>=2*self.step:self.infer()
        self.commit(wait=len(self.cues)>=2*self.step)

    def infer(self):
        if not self.jobs:return
        jobs=self.jobs[:self.step]
        ready=[future.result() for _,_,future in jobs]
        vectors=self.model.embed_inputs(ready)
        if len(vectors)!=len(ready):raise ValueError('SapiensID batch returned an invalid feature count.')
        # Validate every vector before marking a cue complete.
        from .models import normalized_embedding
        vectors=[normalized_embedding(v,4096) for v in vectors]
        for (cue,record,_),vector in zip(jobs,vectors):
            record[2] = vector
            cue['remaining'] -= 1
        del self.jobs[:len(jobs)]
        self.commit()

    @staticmethod
    def _take_crops(cue):
        for record, future in cue.pop('crops', ()):
            record[3], omitted = future.result()
            if omitted:
                record[1] = dict(record[1], crop_status='byte_budget_exceeded')

    def commit(self, *, wait=False):
        while self.cues and self.cues[0]['sealed'] and not self.cues[0]['remaining']:
            cue = self.cues[0]
            crops = cue.get('crops', ())
            if not wait and any(not future.done() for _, future in crops):
                return
            self._take_crops(cue)
            if self.writer is not None:self.writer(self.db,cue['rows'])
            else:
                with self.db:
                    for identifier,metadata,vector,crop in cue['rows']:
                        save(self.db,identifier,self.person_contract,metadata,vector,crop)
            self.cues.popleft()

    def flush(self):
        while self.jobs:self.infer()
        self.commit(wait=True)

    def close(self):
        try:self.stage.close()
        finally:self.jobs.clear();self.cues.clear();self.db=None
