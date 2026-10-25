"""Owned CPU preparation with cross-cue batches and complete-cue publication."""
from collections import deque
from io import BytesIO
from .person_signal import save, contract
from .scan_pipeline import CpuStage


class PreparedCollector:
    def __init__(self, config, model):
        self.config,self.model=config,model
        settings=config.data['execution']
        self.stage=CpuStage(settings['scan_cpu_workers'],settings['scan_queue_bytes']//2)
        self.step=max(1,min(config.data['sapiensid']['execution']['batch_size'],
                            self.stage.limit//(3*384*384*4+384*384+config.data['storage']['max_crop_bytes']+4096)))
        self.cues=deque();self.jobs=[];self.db=None

    def prepare(self,image,value):
        bounds,points=value
        from .sapiensid_v1 import prepare_person_input
        with image.crop(bounds) as crop:
            prepared=prepare_person_input(crop,points)
            encoded=None;omitted=False
            if self.config.data['sapiensid']['storage']['retain_person_crops']:
                output=BytesIO();crop.save(output,format='PNG')
                if output.tell()<=self.config.data['storage']['max_crop_bytes']:encoded=output.getvalue()
                else:omitted=True
        return prepared,encoded,omitted

    def collect(self,db,image,rows,faces):
        if self.db is not None and self.db is not db:raise RuntimeError('Person pipeline database owner changed.')
        self.db=db;cue=dict(rows=[],remaining=0,sealed=False);self.cues.append(cue)
        for row,face in zip(rows,faces):
            if row['rejected'] is not None:continue
            if db.execute('SELECT 1 FROM person_evidence WHERE observation=? AND contract=?',(row['id'],contract(self.config))).fetchone():continue
            metadata=self.model.observation_quality(image,face)
            record=[row['id'],metadata,None,None];cue['rows'].append(record)
            if metadata['identity_rejection'] is None:
                value=self.model.person_input(image,face)
                future=self.stage.submit(self.prepare,image,value,extra=3*384*384*4+384*384+self.config.data['storage']['max_crop_bytes'])
                cue['remaining']+=1;self.jobs.append((cue,record,future))
                if len(self.jobs)>=self.step:self.infer()
        cue['sealed']=True
        if len(self.cues)>=self.step:self.infer()
        self.commit()

    def infer(self):
        if not self.jobs:return
        ready=[future.result() for _,_,future in self.jobs]
        vectors=self.model.embed_inputs([parts[0] for parts in ready])
        if len(vectors)!=len(ready):raise ValueError('SapiensID batch returned an invalid feature count.')
        # Validate every vector before marking a cue complete.
        from .models import normalized_embedding
        vectors=[normalized_embedding(v,4096) for v in vectors]
        for (cue,record,_),vector,(_,crop,omitted) in zip(self.jobs,vectors,ready):
            record[2:]=vector,crop
            if omitted:record[1]=dict(record[1],crop_status='byte_budget_exceeded')
            cue['remaining']-=1
        self.jobs.clear();self.commit()

    def commit(self):
        while self.cues and self.cues[0]['sealed'] and not self.cues[0]['remaining']:
            cue=self.cues[0]
            with self.db:
                for identifier,metadata,vector,crop in cue['rows']:
                    save(self.db,identifier,contract(self.config),metadata,vector,crop)
            self.cues.popleft()

    def flush(self):
        self.infer();self.commit()

    def close(self):
        try:self.stage.close()
        finally:self.jobs.clear();self.cues.clear();self.db=None
