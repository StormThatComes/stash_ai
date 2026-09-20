"""Shared offline helpers; no test collection or fixture registration."""


class Tagger:
    def __init__(self,settings,labels,scores):
        self.settings,self.labels,self.scores=settings,labels,scores
        self.batch_size,self.offset,self.closed=settings.batch_size,0,False
    def prepare(self,image): return 0
    def predict_batch(self,batch):
        result=self.scores[self.offset:self.offset+len(batch)];self.offset+=len(batch);return result
    def close(self): self.closed=True


class IndexedTagger:
    def __init__(self,settings,scores,labels,fail_at=None,error=KeyboardInterrupt):
        self.settings,self.scores,self.labels=settings,scores,labels
        self.batch_size=settings.batch_size
        self.fail_at,self.error=fail_at,error
        self.seen,self.prepared=[],[]
        self.closed=False
    def prepare(self,image): self.prepared.append(image);return image
    def predict_batch(self,images):
        if self.fail_at is not None and min(images)>=self.fail_at: raise self.error("interrupted")
        self.seen.extend(images)
        return self.scores[images]
    def close(self): self.closed=True
