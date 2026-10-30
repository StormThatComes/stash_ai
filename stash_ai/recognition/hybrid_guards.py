"""Original merged cores veto later merges without acquiring reference authority."""
from .resource_errors import ResourceLimit
from collections import defaultdict
from dataclasses import replace
import json
import numpy as np
from .disk_body_modes import resident_bytes

from .database import encoded
from .person_signal import contract
from .scan_heads import predicate as current_scan


def retain(db,config,scope,namespace,identities):
    from .identity_merges import canonical
    from .clustering import reference_admission,vector
    from .reference_compatibility import fallback_sql
    from .scanning import scan_identity
    from .person_anchor import anchor_gate
    core_ids=defaultdict(set)
    owned={i.person:i for i in identities};guards=defaultdict(set);body_guards=defaultdict(set);proof_bytes=0
    body_route_faces=defaultdict(set)
    for row in db.execute('SELECT a.target,a.evidence FROM identity_merges a JOIN people p ON p.id=a.target '
                          'WHERE p.scope=? AND p.namespace=?',(scope,namespace)):
        root=canonical(db,row['target'])
        if root not in owned:continue
        proof_bytes+=len(row['evidence'].encode())
        if proof_bytes>config.data['sapiensid']['execution']['plan_bytes']:
            raise ResourceLimit('SapiensID plan_bytes budget exceeded by original core history.')
        proof=json.loads(row['evidence'])
        if proof.get('route') in ('body_global','body_scene'):
            ids={i for values in proof.get('face_witnesses',{}).values() for i in values}
            # Older promotions did not persist an explicit face witness list.
            body_ids=list({i for values in proof.get('cores',{}).values() for i in values})
            ids.update(r[0] for r in db.execute('SELECT face FROM body_observations WHERE face IS NOT NULL '
                'AND id IN (SELECT value FROM json_each(?))',(encoded(body_ids),)))
            guards[root].update(ids);body_route_faces[root].update(ids)
            continue
        if proof.get('route')!='sapiensid_hybrid':continue
        for pair in proof['proofs']:
            for observations in pair['observations']+pair.get('guard_observations',[]):
                guards[root].update(observations)
            body_groups=pair.get('body_witness_observations') or pair.get('person_observations') or pair['observations']
            for face_ids,person_ids in zip(pair['observations'],body_groups):
                core_ids[root].add((tuple(face_ids),tuple(person_ids)))
            for observations in pair.get('body_witness_observations',body_groups)+pair.get('guard_person_observations',pair.get('guard_observations',[])):
                body_guards[root].update(observations)
        if any('body_witness_observations' not in pair for pair in proof['proofs']):
            # Old proofs capped representatives. Retained source attachments
            # recover omitted vetoes without acquiring reference authority.
            from .reference_roles import observation_sql
            sources=proof.get('original_cores',proof.get('people',[]))
            body_guards[root].update(r[0] for r in db.execute(
                'SELECT DISTINCT o.id FROM observations o JOIN candidates c ON o.id='+observation_sql('c')+
                ' JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
                "WHERE json_extract(c.payload,'$.hybrid_merge.source') IN (SELECT value FROM json_each(?)) "
                "AND e.status='ready'",(contract(config),encoded(sources))))
    admission=reference_admission(config.data['clustering'],scan_policy=scan_identity(config,namespace),
                                   compatibility=fallback_sql(db,config,namespace))
    expected=contract(config);result=[]
    used=2*sum(i.faces.nbytes+i.persons.nbytes+sum(resident_bytes(m) for m in i.body_modes) for i in identities)
    for identity in identities:
        ids=sorted(guards[identity.person]-set(identity.observations))
        body_ids=sorted(body_guards[identity.person]-set(identity.person_observations))
        faces=[];persons=[];missing=[];face_missing=[]
        face_map=dict(zip(identity.observations,identity.faces))
        body_map=dict(zip(identity.person_observations,identity.persons))
        wanted=sorted(set(ids)|set(body_ids)|set(identity.person_observations))
        retained={row['id']:row for row in db.execute(
            'SELECT o.id,o.scan,o.cue,o.seconds,o.quality,o.dimensions,o.embedding,e.status AS person_status,e.embedding AS person_embedding,e.metadata AS person_metadata FROM observations o '
            'JOIN scans s ON s.id=o.scan JOIN members m ON m.observation=o.id '
            'LEFT JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
            "WHERE o.id IN (SELECT value FROM json_each(?)) AND m.person=? AND s.scope=? AND s.namespace=? AND s.status='complete' "
            'AND o.rejected IS NULL AND (('+admission+') OR (o.id IN (SELECT value FROM json_each(?)) AND '+
            current_scan()+'))',
            (expected,encoded(wanted),identity.person,scope,namespace,encoded(sorted(body_route_faces[identity.person]))))} if wanted else {}
        for identifier in wanted:
            row=retained.get(identifier)
            if row is None:
                missing.append(identifier)
                if identifier in ids:face_missing.append(identifier)
                continue
            if identifier in ids:
                face=vector(row);faces.append(face);face_map[identifier]=face;used+=2*face.nbytes
            if identifier in body_ids:
                if row['person_status']!='ready':
                    missing.append(identifier);continue
                person=np.frombuffer(row['person_embedding'],dtype='<f4')
                if person.shape!=(4096,) or not np.isfinite(person).all() or abs(float(np.linalg.norm(person))-1)>.001:
                    raise ValueError('Invalid original SapiensID core feature: '+identifier)
                persons.append(person);used+=2*person.nbytes
                # Original evidence keeps veto power even below today's size
                # floor, but cannot become an anchor for query recovery.
                if anchor_gate(row['person_metadata'],config.data['sapiensid']['identities'],
                        collection=config.data['sapiensid']['collection'])['eligible']:
                    body_map[identifier]=person
            if used>config.data['sapiensid']['execution']['reference_bytes']:
                raise ResourceLimit('SapiensID reference_bytes budget exceeded by original cores.')
        cores=[];core_modes=[]
        for face_ids,person_ids in sorted(core_ids[identity.person]):
            if not face_ids or any(i not in face_map for i in face_ids):continue
            # Keep original endpoint provenance; never pool unrelated cores.
            from .independent_evidence import body_vote,Votes
            from .hybrid_modes import body_modes
            admitted=[i for i in person_ids if i in body_map and i in retained]
            votes=Votes()
            records=[]
            for identifier in admitted:
                row=body_vote(retained[identifier]);votes.add(row)
                records.append(dict(row,embedding=body_map[identifier].tobytes()))
            modes=(body_modes(records,config) if len(votes)>=config.data['sapiensid']['matching']['min_support']
                   and not any(i in missing for i in person_ids) else ())
            cores.append((face_ids,np.asarray([face_map[i] for i in face_ids]),
                          np.asarray([body_map[i] for i in admitted]) if modes else np.empty((0,4096))))
            core_modes.append(modes)
        used+=sum(f.nbytes+b.nbytes for _,f,b in cores)+sum(m.nbytes for modes in core_modes for m in modes)
        if used>config.data['sapiensid']['execution']['reference_bytes']:
            raise ResourceLimit('SapiensID reference_bytes budget exceeded by query cores.')
        result.append(replace(identity,guard_face_missing=tuple(face_missing),query_cores=tuple(cores),query_core_modes=tuple(core_modes),guard_observations=tuple(ids),guard_missing=tuple(missing),
            guard_faces=np.asarray(faces) if faces else None,guard_persons=np.asarray(persons) if persons else None,
            guard_person_observations=tuple(body_ids)))
    return tuple(result)
