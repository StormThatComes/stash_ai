"""One deterministic cue allowance for contact, mask and second-pose retries."""


class Exhausted(Exception):pass


class Budget:
    def __init__(self,image,settings):
        self.remaining=settings.get('max_retry_jobs_per_cue',16)
        source=getattr(image,'_recognition_pose_source',None)
        signature=getattr(image,'_recognition_cue_signature',None)
        if source and signature:
            total=max(1,len(source[0].cues));cue=signature[0]
            quota,remainder=divmod(settings.get('max_retry_jobs_per_scene',4096),total)
            self.remaining=min(self.remaining,quota+int(cue<remainder))
        self.used={};self.skipped={}

    def admit(self,stage,amount=1,record=None):
        from .profiling import pipeline_count
        if amount>self.remaining:
            self.skipped[stage]=self.skipped.get(stage,0)+amount
            pipeline_count('refinement_budget_skipped_'+stage,amount)
            if record is not None:record['retry_skip_reason']='shared_retry_budget'
            return False
        self.remaining-=amount;self.used[stage]=self.used.get(stage,0)+amount
        pipeline_count('refinement_retry_jobs_'+stage,amount)
        return True


def begin(frames,settings):
    for image,*_ in frames:image._recognition_retry_budget=Budget(image,settings)


def budget(image,settings):
    current=getattr(image,'_recognition_retry_budget',None)
    if current is None:
        current=image._recognition_retry_budget=Budget(image,settings)
    return current
