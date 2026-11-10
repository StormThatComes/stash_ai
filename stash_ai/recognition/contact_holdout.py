"""Score visible VTT people and one controlled association fault in private replay."""
from collections import Counter, defaultdict
import json

from .body_observations import available, contract, repair_projections
from .crowd_labels import torso_owner
from .reference_roles import STRICT_REASONS


def physical_inputs(db, config, scans, visible, labels, locations):
    """Keep missed people as empty addresses. Human labels supersede geometry."""
    groups = {}
    swap_cases = []
    annotations = defaultdict(list)
    for oid,(scene,cue) in locations.items():
        annotations[scene,cue,labels[oid]].append(oid)
    for scan in scans:
        for number,people in sorted(visible.get(scan['scene'],{}).items(),key=lambda x:int(x[0])):
            cue = int(number)
            pairs = []
            manual = {p['truth'] for p in people if p.get('binding') is not None}
            for label in people:
                key = scan['scene'],cue,label['truth']
                groups[key] = list(annotations[key])
            if not available(db):
                continue
            for oid,geometry,face,rejected,ready in db.execute(
                    'SELECT id,geometry,face,rejected,embedding IS NOT NULL FROM body_observations '
                    'WHERE scan=? AND cue=? AND contract=? ORDER BY id',(scan['id'],cue,contract(config))):
                truth = labels.get(oid) or torso_owner(json.loads(geometry),people)
                if (scan['scene'],cue,truth) not in groups or (truth in manual and oid not in labels):
                    continue
                group = groups[scan['scene'],cue,truth]
                if oid not in group:
                    group.append(oid)
                if rejected is None and ready:
                    pairs.append((oid,truth,face))
            if (len(people)==len(pairs)==2 and pairs[0][1]!=pairs[1][1]
                    and pairs[0][2]!=pairs[1][2] and any(row[2] for row in pairs)
                    and all(not face or labels.get(face,truth)==truth for _,truth,face in pairs)):
                swap_cases.append((scan['id'],cue,pairs[0][0],pairs[1][0]))
    return groups, swap_cases


def stress_cases(candidates, visible, scans, samples, locations):
    # Select separate contact episodes, prioritizing different physical events.
    scenes = {scan['id']: scan['scene'] for scan in scans}
    backs = {(scene, cue) for oid,(scene,cue) in locations.items() if 'back' in samples[oid]['slices']}
    episodes_by_cue = {}
    for scene, labels in visible.items():
        start = previous = None
        for number,people in sorted(labels.items(),key=lambda item:int(item[0])):
            cue = int(number)
            if len(people)==2:
                if previous is None or cue!=previous+1:
                    start = cue
                episodes_by_cue[scene,cue] = start
                previous = cue
            else:
                previous = None
    options = []
    for case in candidates:
        scan,cue,*_ = case
        scene = scenes[scan]
        labels = visible.get(scene,{})
        if (scene,cue+1) in backs:
            kind = 'before_turn'
        elif (scene,cue-1) in backs:
            kind = 'after_turn'
        elif len(labels.get(str(cue+1), ())) == 1:
            kind = 'partner_exit'
        else:
            kind = 'contact'
        start = episodes_by_cue[scene,cue]
        options.append((kind,(scene,start),case))
    chosen, episodes, kinds = [], set(), set()
    for diverse in (True,False):
        for kind,episode,case in sorted(options, key=lambda x:(x[0]=='contact',x[1],x[2][1])):
            if episode in episodes or diverse and kind in kinds:
                continue
            chosen.append((kind,case))
            kinds.add(kind)
            episodes.add(episode)
            if len(chosen)==3:
                return chosen
    return chosen


def visible_scores(groups, decisions, labels, references=(), bindings=None):
    counts = Counter(dict.fromkeys(('correct','partner','fragment','withheld','missing','unknown_reference','split_reference'),0))
    roots = defaultdict(set)
    for oid,truth in labels.items():
        d = decisions[oid]
        if (d.get('evidence_kind')!='person' and d['accepted_identity'] and
                d.get('evidence',{}).get('reason') in STRICT_REASONS):
            roots[truth].add(d['accepted_identity'])
    designated = defaultdict(set)
    for oid in references:
        truth = labels[oid]
        decision = decisions[oid]
        owner = decision['accepted_identity']
        if owner in roots[truth] and decision.get('evidence_kind') != 'person' and (decision.get('evidence') or {}).get('reason') in STRICT_REASONS:
            designated[truth].add(owner)
    owners = defaultdict(set)
    for truth,ids in roots.items():
        for owner in ids:
            owners[owner].add(truth)
    for location,addresses in groups.items():
        truth = location[2]
        ids = {decisions[oid]['accepted_identity'] for oid in addresses} - {None}
        if not addresses:
            result = (bindings or {}).get(location,{}).get('binding','missing')
            if result=='bound':
                result = 'missing'
        elif not ids:
            result = 'withheld'
        elif any(owners[p]-{truth} for p in ids):
            result = 'partner'
        elif len(roots[truth]) > 1 and len(designated[truth]) != 1:
            result = 'split_reference'
        elif not roots[truth]:
            result = 'unknown_reference'
        else:
            result = 'correct' if ids==(designated[truth] or roots[truth]) else 'fragment'
        counts[result] += 1
    return dict(total=len(groups),reference_identities=sum(bool(ids) for ids in roots.values()),
                split_identities=sum(len(ids)>1 for ids in roots.values()),**counts)


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


def spread(before, after, locations, scene, cue, labels):
    def signatures(rows):
        members, roots = defaultdict(set),defaultdict(set)
        for oid,place in locations.items():
            decision = rows[oid]
            owner = decision['accepted_identity']
            if owner is not None:
                members[owner].add(oid)
                if (place!=(scene,cue) and decision.get('evidence_kind')!='person' and
                        decision.get('evidence',{}).get('reason') in STRICT_REASONS):
                    roots[owner].add(labels[oid])
        return members,roots
    original, first_roots = signatures(before)
    stressed, last_roots = signatures(after)
    changed,ambiguous,partitions = set(),set(),set()
    observations = 0
    for oid,place in locations.items():
        a,b = before[oid]['accepted_identity'],after[oid]['accepted_identity']
        if original.get(a)!=stressed.get(b):
            partitions.add(place)
        if a is None or b is None:
            moved = a!=b
        elif first_roots[a] and last_roots[b]:
            moved = first_roots[a]!=last_roots[b]
        elif a==b or original[a]==stressed[b]:
            moved = False
        else:
            # A renamed unanchored fragment cannot establish a physical ID switch.
            ambiguous.add(place)
            continue
        if moved:
            observations += 1
            changed.add(place)
    later = [c for s,c in changed if s==scene and c>cue]
    return dict(changed_cues=len(changed),changed_observations=observations,later_cues=len(later),
                last_later_cue=max(later,default=None),other_scenes=sorted({s for s,_ in changed if s!=scene}),
                partition_changed_cues=len(partitions),unresolved_alignment_cues=len(ambiguous))
