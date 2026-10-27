"""Explicit, bounded full-replay neighbourhood validation."""
from .merge_report_policy import parameter
from .merge_solver_metrics import scale


def probes(config, changes, active, limit=4):
    keys=list(dict.fromkeys(list(changes)+sorted(active,key=active.get,reverse=True)))
    yielded=0
    for key in keys:
        current=changes.get(key,parameter(config.data,key))
        if type(current) not in (int,float):continue
        step=1 if key.endswith('support') else .002*scale(key,current)
        for sign in (-1,1):
            value=round(max(1 if key.endswith('support') else 0,current+sign*step),6)
            if key.endswith('support'):value=int(value)
            if value==current:continue
            yield dict(changes,**{key:value})
            yielded+=1
            if yielded>=limit:return


def confidence(checks):
    # Positive preference requires at least two full replays with no loss.
    return int(len(checks)>=2 and all(checks))
