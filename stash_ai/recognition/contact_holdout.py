"""Score visible VTT people and one controlled association fault in private replay."""
from collections import Counter, defaultdict
import json

from .body_observations import available, contract, repair_projections
from .crowd_labels import torso_owner
from .reference_roles import STRICT_REASONS


def physical_inputs(db, config, scans, visible, labels, locations):
    """Keep missed people as empty addresses. Human labels supersede geometry."""
    groups = {}
    swap_case = None
    annotations = defaultdict(list)
    for oid,(scene,cue) in locations.items():
        annotations[scene,cue,labels[oid]].append(oid)
    for scan in scans:
        for number,people in sorted(visible.get(scan['scene'],{}).items(),key=lambda x:int(x[0])):
            cue = int(number)
            pairs = []
            for label in people:
                key = scan['scene'],cue,label['truth']
                groups[key] = list(annotations[key])
            if not available(db):
                continue
            for oid,geometry,face,rejected,ready in db.execute(
                    'SELECT id,geometry,face,rejected,embedding IS NOT NULL FROM body_observations '
                    'WHERE scan=? AND cue=? AND contract=? ORDER BY id',(scan['id'],cue,contract(config))):
                truth = labels.get(oid) or torso_owner(json.loads(geometry),people)
                if (scan['scene'],cue,truth) not in groups:
                    continue
                group = groups[scan['scene'],cue,truth]
                if oid not in group:
                    group.append(oid)
                if rejected is None and ready:
                    pairs.append((oid,truth,face))
            if (swap_case is None and len(people)==len(pairs)==2 and pairs[0][1]!=pairs[1][1]
                    and pairs[0][2]!=pairs[1][2] and any(row[2] for row in pairs)
                    and all(not face or labels.get(face,truth)==truth for _,truth,face in pairs)):
                swap_case = scan['id'],cue,pairs[0][0],pairs[1][0]
    return groups,swap_case


def visible_scores(groups, decisions, labels):
    counts = Counter(dict.fromkeys(('correct','partner','fragment','withheld','missing','unknown_reference'),0))
    roots = defaultdict(set)
    for oid,truth in labels.items():
        d = decisions[oid]
        if (d.get('evidence_kind')!='person' and d['accepted_identity'] and
                d.get('evidence',{}).get('reason') in STRICT_REASONS):
            roots[truth].add(d['accepted_identity'])
    owners = defaultdict(set)
    for truth,ids in roots.items():
        for owner in ids:
            owners[owner].add(truth)
    for (_,_,truth),addresses in groups.items():
        ids = {decisions[oid]['accepted_identity'] for oid in addresses} - {None}
        if not addresses:
            result = 'missing'
        elif not ids:
            result = 'withheld'
        elif any(owners[p]-{truth} for p in ids):
            result = 'partner'
        elif len(roots[truth])!=1:
            result = 'unknown_reference'
        else:
            result = 'correct' if ids==roots[truth] else 'fragment'
        counts[result] += 1
    return dict(total=len(groups),**counts)


def swap(db,config,scan,cue,first,second):
    from .person_signal import contract as projection_contract
    faces = dict(db.execute('SELECT id,face FROM body_observations WHERE scan=? AND cue=? AND id IN (?,?)',
                            (scan,cue,first,second)))
    if set(faces)!={first,second} or faces[first]==faces[second]:
        raise ValueError('Contact stress requires two distinct body associations in one cue.')
    db.execute('DELETE FROM person_evidence WHERE contract=? AND observation IN (?,?)',
               (projection_contract(config),faces[first],faces[second]))
    db.executemany('UPDATE body_observations SET face=? WHERE id=?',[(faces[first],second),(faces[second],first)])
    repair_projections(db,config,scan,cue=cue)


def spread(before, after, locations, scene, cue):
    # Compare membership sets, invariant to UUID renaming between private replays.
    def signatures(rows):
        owners = defaultdict(set)
        for oid in locations:
            owner = rows[oid]['accepted_identity']
            if owner:
                owners[owner].add(oid)
        banks = {p:frozenset(ids) for p,ids in owners.items()}
        return {oid:banks.get(rows[oid]['accepted_identity']) for oid in locations}
    original,stressed = signatures(before),signatures(after)
    changed = {place for oid,place in locations.items() if original[oid]!=stressed[oid]}
    later = [c for s,c in changed if s==scene and c>cue]
    return dict(changed_cues=len(changed),later_cues=len(later),last_later_cue=max(later,default=None),
                other_scenes=sorted({s for s,_ in changed if s!=scene}))
