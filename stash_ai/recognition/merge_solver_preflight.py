"""Frozen-evidence ordering only; never certifies or rejects a replay candidate."""
from itertools import islice
from .merge_report_policy import parameter


def cells(values, *, integer=False):
    """Represent open intervals between measured boundaries, without grid growth."""
    values = sorted({v for v in values if type(v) in (int,float)})
    for left,right in zip(values, values[1:]):
        middle = (left+right)//2 if integer else round((left+right)/2,6)
        if left < middle < right:
            yield middle


def boundary_probes(changes, options):
    for key,values in options.items():
        for value in cells(values,integer=key.endswith('support')):
            yield dict(changes,**{key:value})


def estimate(config, changes, feedback):
    query = feedback.get('query',{})
    positive = query.get('positive_witnesses',())
    negative = query.get('negative_witnesses',())
    def accepts(witness):
        return all(changes.get(k,parameter(config.data,k))<=v for k,v in witness.items())
    # Incomplete witnesses are useful for ordering, never for declaring safety.
    return (sum(not accepts(w) for w in negative)/max(1,len(negative)),
            sum(accepts(w) for w in positive)/max(1,len(positive)))


def ordered(config, candidates, feedback, window=8):
    """Bounded lookahead; emit every input. Independent search lanes stay intact."""
    source=iter(candidates)
    while batch:=list(islice(source,window)):
        yield from sorted(batch,key=lambda p:estimate(config,p,feedback),reverse=True)
