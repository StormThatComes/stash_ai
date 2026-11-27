"""Deterministic bounded force layout. No Tk objects or recognition decisions."""
import hashlib
import math


def initial(ids, previous=None):
    previous = previous or {}
    result = {}
    for identifier in ids:
        seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:12],16)
        angle = seed % 100000 / 100000 * math.tau
        radius = 100 + (seed // 100000) % 250
        result[identifier] = previous.get(identifier, (math.cos(angle)*radius,math.sin(angle)*radius))
    return result


def step(positions, edges, pinned=()):
    import numpy as np
    ids = list(positions)
    if not ids: return {}, 0.
    index = {k:i for i,k in enumerate(ids)}
    xy = np.asarray([positions[k] for k in ids], dtype=float)
    delta = xy[:,None,:]-xy[None,:,:]
    distance2 = np.maximum((delta*delta).sum(axis=2),100.)
    force = (delta*(350./distance2)[:,:,None]).sum(axis=1) - xy*.002
    for edge in edges:
        a,b = index[edge['source']],index[edge['target']]
        d = xy[b]-xy[a]; distance = max(float(np.linalg.norm(d)),1.)
        strength = .015 if edge['relation'] in ('similarity','rival') else .045
        spring = d*(distance-85.)/distance*strength
        force[a] += spring; force[b] -= spring
    force = np.clip(force*.45,-6.,6.)
    for identifier in pinned:
        if identifier in index: force[index[identifier]]=0
    moved = xy+force
    return {k:tuple(map(float,moved[i])) for i,k in enumerate(ids)}, float(np.max(np.abs(force)))


def node_role(node):
    """The displayed membership role, or the recorded View node itself."""
    if node['type'] == 'view':
        return 'view'
    if node['type'] == 'observation':
        state = node.get('state')
        return state if state in ('anchor', 'auxiliary', 'attachment', 'reference') else 'unknown'
    return None


def project(graph, expanded):
    """Presentation groups do not assert production Views or new memberships."""
    nodes = {n['id']:n for n in graph['nodes']}
    groups = {}
    titles = {}
    for n in nodes.values():
        if n['type']=='observation' and n.get('current_member'):
            authority = n.get('authority', 'unknown')
            group = f"group:{n['scene']}:{authority}"
            groups[n['id']] = group
            titles[group] = f"Scene {n['scene']} · {authority.replace('_', ' ')}"
    for e in graph['edges']:
        if e['relation']=='assigned' and e['target'] in groups:
            groups[e['source']] = groups[e['target']]
    # A recorded View owns its query decisions in the presentation. Shared
    # source witnesses retain their own groups and their explicit support edges.
    for e in graph['edges']:
        if e['relation']=='view_match':
            group='group:'+e['source'];groups[e['source']]=group;groups[e['target']]=group
            titles[group] = 'View matches'
            for a in graph['edges']:
                if a['source']==e['target'] and a['relation']=='assigned':groups[a['target']]=group
    mapped, output = {}, {}
    for identifier,n in nodes.items():
        group=groups.get(identifier)
        shown=group if group and group not in expanded else identifier
        mapped[identifier]=shown
        if shown==identifier:output[shown]=n
        else:
            if shown not in output:
                output[shown] = dict(id=shown, type='group', label=titles[group], members=[],
                                     role_counts={}, provenance='presentation_group')
            entry = output[shown]
            entry['members'].append(identifier)
            role = node_role(n)
            if role:
                counts = entry['role_counts']
                counts[role] = counts.get(role, 0) + 1
    edges={}
    for e in graph['edges']:
        a,b=mapped[e['source']],mapped[e['target']]
        if a!=b:
            entry=edges.setdefault((a,b,e['relation']),dict(e,source=a,target=b,count=0))
            entry['count']+=1
    return dict(nodes=list(output.values()),edges=list(edges.values())), set(groups.values())
