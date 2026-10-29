"""Original merged cores veto later merges without acquiring reference authority."""
from collections import defaultdict
from dataclasses import replace
import json
import numpy as np

from .database import encoded
from .person_signal import contract


def retain(db,config,scope,namespace,identities):
    from .identity_merges import canonical
    from .clustering import reference_admission,vector
    from .reference_compatibility import fallback_sql
    from .scanning import scan_identity
    from .person_anchor import pixel_gate
    core_ids=defaultdict(set)
    owned={i.person:i for i in identities};guards=defaultdict(set);body_guards=defaultdict(set);proof_bytes=0
    for row in db.execute('SELECT a.target,a.evidence FROM identity_merges a JOIN people p ON p.id=a.target '
                          'WHERE p.scope=? AND p.namespace=?',(scope,namespace)):
        root=canonical(db,row['target'])
        if root not in owned:continue
        proof_bytes+=len(row['evidence'].encode())
        if proof_bytes>config.data['sapiensid']['execution']['plan_bytes']:
            raise ValueError('SapiensID plan_bytes budget exceeded by original core history.')
        proof=json.loads(row['evidence'])
        if proof.get('route')!='sapiensid_hybrid':continue
        for pair in proof['proofs']:
            for observations in pair['observations']+pair.get('guard_observations',[]):
                guards[root].update(observations)
            body_groups=pair.get('person_observations') or pair['observations']
            for face_ids,person_ids in zip(pair['observations'],body_groups):
                core_ids[root].add((tuple(face_ids),tuple(person_ids)))
            for observations in body_groups+pair.get('guard_person_observations',pair.get('guard_observations',[])):
                body_guards[root].update(observations)
    admission=reference_admission(config.data['clustering'],scan_policy=scan_identity(config,namespace),
                                   compatibility=fallback_sql(db,config,namespace))
    expected=contract(config);result=[]
    used=2*sum(i.faces.nbytes+i.persons.nbytes+sum(m.nbytes for m in i.body_modes) for i in identities)
    for identity in identities:
        ids=sorted(guards[identity.person]-set(identity.observations))
        body_ids=sorted(body_guards[identity.person]-set(identity.person_observations))
        faces=[];persons=[];missing=[];face_missing=[]
        face_map=dict(zip(identity.observations,identity.faces))
        body_map=dict(zip(identity.person_observations,identity.persons))
        wanted=sorted(set(ids)|set(body_ids))
        retained={row['id']:row for row in db.execute(
            'SELECT o.id,o.dimensions,o.embedding,e.status AS person_status,e.embedding AS person_embedding,e.metadata AS person_metadata FROM observations o '
            'JOIN scans s ON s.id=o.scan JOIN members m ON m.observation=o.id '
            'LEFT JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
            "WHERE o.id IN (SELECT value FROM json_each(?)) AND m.person=? AND s.scope=? AND s.namespace=? AND s.status='complete' "
            'AND o.rejected IS NULL AND '+admission,
            (expected,encoded(wanted),identity.person,scope,namespace))} if wanted else {}
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
                if pixel_gate(row['person_metadata'],config.data['sapiensid']['identities'])['eligible']:
                    body_map[identifier]=person
            if used>config.data['sapiensid']['execution']['reference_bytes']:
                raise ValueError('SapiensID reference_bytes budget exceeded by original cores.')
        cores=[]
        for face_ids,person_ids in sorted(core_ids[identity.person]):
            if not face_ids or any(i not in face_map for i in face_ids):continue
            # Keep original endpoint provenance; never pool unrelated cores.
            admitted=[body_map[i] for i in person_ids if i in body_map]
            cores.append((face_ids,np.asarray([face_map[i] for i in face_ids]),
                          np.asarray(admitted) if admitted and not any(i in missing for i in person_ids) else np.empty((0,4096))))
        used+=sum(f.nbytes+b.nbytes for _,f,b in cores)
        if used>config.data['sapiensid']['execution']['reference_bytes']:
            raise ValueError('SapiensID reference_bytes budget exceeded by query cores.')
        result.append(replace(identity,guard_face_missing=tuple(face_missing),query_cores=tuple(cores),guard_observations=tuple(ids),guard_missing=tuple(missing),
            guard_faces=np.asarray(faces) if faces else None,guard_persons=np.asarray(persons) if persons else None,
            guard_person_observations=tuple(body_ids)))
    return tuple(result)
