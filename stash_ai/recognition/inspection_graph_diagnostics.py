"""Explicit graph exploration; reachability is not a counterfactual replay."""
from collections import deque
from contextlib import closing
import json
from .inspection_runtime import open_readonly, stamp, check_cancelled
from .identity_inspection import TABLES
from .models import namespace


def related(graph, selected, *, downstream=False):
    adjacency = {}
    for e in graph['edges']:
        if e['relation'] in ('similarity','rival'):continue
        a,b=(e['source'],e['target']) if downstream else (e['target'],e['source'])
        adjacency.setdefault(a,set()).add(b)
    seen={selected};queue=deque([selected])
    while queue:
        check_cancelled()
        for other in adjacency.get(queue.popleft(),()):
            if other not in seen:seen.add(other);queue.append(other)
    return seen


def bridges(graph):
    """Undirected structural bottlenecks among retained positive evidence only."""
    adjacency={n['id']:set() for n in graph['nodes']}
    for e in graph['edges']:
        if e['relation'] in ('similarity','rival'):continue
        adjacency[e['source']].add(e['target']);adjacency[e['target']].add(e['source'])
    order,low,found={}, {}, []
    def visit(node,parent=None):
        check_cancelled();order[node]=low[node]=len(order)
        for other in sorted(adjacency[node]):
            if other==parent:continue
            if other not in order:
                visit(other,node);low[node]=min(low[node],low[other])
                if low[other]>order[node] and len(adjacency[node])>1 and len(adjacency[other])>1:
                    found.append((node,other))
            else:low[node]=min(low[node],order[other])
    for node in adjacency:
        if node not in order:visit(node)
    return found


def similarities(reader, graph, *, threshold=.7, neighbours=3):
    """Cosine edges only within modality on this bounded page, excluding same cue.

    Neither connected components nor these new edges are production Views.
    """
    import numpy as np
    if not -1<=threshold<=1 or not 1<=neighbours<=10:raise ValueError('Invalid similarity options.')
    revision=stamp(reader.config.database)
    if revision!=graph['revision']:raise ValueError('Recognition changed. Refresh the graph.')
    output=[];missing=0
    with closing(open_readonly(reader.config.database)) as db:
        for kind,(table,_) in TABLES.items():
            candidates=[n for n in graph['nodes'] if n.get('kind')==kind and n.get('current_member')]
            vectors=[];nodes=[];pixels=[]
            for n in candidates:
                check_cancelled()
                row=db.execute(f'SELECT o.embedding,o.quality FROM {table} o JOIN scans s ON s.id=o.scan '
                    'WHERE o.id=? AND s.scope=? AND s.namespace=?',
                    (n['observation'],graph['person']['scope'],namespace(reader.config))).fetchone()
                if not row or not row['embedding']:
                    missing+=1;continue
                vector=np.frombuffer(row['embedding'],dtype='<f4')
                norm=float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or norm<=0:
                    missing+=1;continue
                vectors.append(vector/norm);nodes.append(n)
                pixels.append(json.loads(row['quality']).get('evidence_pixel_sha256'))
            if not vectors:continue
            if len({len(v) for v in vectors})!=1:raise ValueError('Incompatible retained embedding dimensions.')
            scores=np.asarray(vectors)@np.asarray(vectors).T
            selected={}
            for i,n in enumerate(nodes):
                check_cancelled()
                eligible=[j for j,o in enumerate(nodes) if i!=j and
                    (n['scene'],n['cue'])!=(o['scene'],o['cue']) and
                    not (pixels[i] and pixels[i]==pixels[j]) and scores[i,j]>=threshold]
                for j in sorted(eligible,key=lambda j:(-float(scores[i,j]),nodes[j]['id']))[:neighbours]:
                    a,b=sorted((n['id'],nodes[j]['id']))
                    selected[a,b]=dict(source=a,target=b,relation='similarity',score=float(scores[i,j]),
                        threshold=threshold,provenance='new_comparison',kind=kind)
            output.extend(selected.values())
    if revision!=stamp(reader.config.database):raise ValueError('Recognition changed during comparisons. Refresh.')
    return dict(edges=output,missing=missing)
