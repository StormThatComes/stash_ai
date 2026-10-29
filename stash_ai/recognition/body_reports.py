"""Typed body evidence shared by offline consumers; never fabricate face rows."""
import json
from collections import Counter

from .body_identity import available, enabled, policy, anchor
from .body_observations import contract, coverage
from .database import encoded
from .person_anchor import pixel_gate


def observations(db,config,scans):
    if not enabled(config) or not available(db):return {}
    return dict(db.execute('SELECT b.id,s.scene FROM body_observations b JOIN scans s ON s.id=b.scan '
        'WHERE b.scan IN (SELECT value FROM json_each(?)) AND b.contract=?',(encoded(list(scans)),contract(config))))


def members(db,config=None):
    if not available(db) or config is not None and not enabled(config):return {}
    return dict(db.execute("SELECT observation,person FROM body_members WHERE status='accepted'"))


def goal_rows(db,config,person):
    if not enabled(config) or not available(db):return []
    rows=[]
    for raw in db.execute('SELECT b.id,b.scan,s.scene,b.cue,b.seconds,b.face,b.quality,b.rejected,s.status,m.role '
        "FROM body_members m JOIN body_observations b ON b.id=m.observation JOIN scans s ON s.id=b.scan "
        "WHERE m.person=? AND m.status='accepted' AND b.contract=? "
        "AND NOT EXISTS (SELECT 1 FROM members f WHERE f.observation=b.face AND f.person=m.person) ORDER BY b.id",(person,contract(config))):
        row=dict(raw);strong=row['role']=='anchor' and anchor(row,config.data['sapiensid']['identities'])
        eligible=row['status']=='complete' and row['rejected'] is None
        metadata=json.loads(row['quality'])
        fingerprint=metadata.get('quality',{}).get('pixel_digest')
        if fingerprint:metadata['evidence_pixel_sha256']=fingerprint
        row['quality']=encoded(metadata)
        row.update(evidence_kind='person',reference_authority=strong,
            goal_role='reference' if eligible and strong else 'uncertain',
            sample_role='anchor' if eligible and strong else 'auxiliary' if eligible else 'other',body_state='ready')
        rows.append(row)
    return rows


def report(db,config,scans):
    result=coverage(db,config,scans);counts=Counter();reasons=Counter();identities=[]
    if available(db):
        for row in db.execute('SELECT m.status,m.role,m.proof FROM body_members m JOIN body_observations b ON b.id=m.observation '
            'WHERE b.scan IN (SELECT value FROM json_each(?)) AND b.contract=?',(encoded(list(scans)),contract(config))):
            counts[row['status']]+=1;counts[row['role']]+=1
            reasons[json.loads(row['proof']).get('reason','unknown')]+=1
        identities=[dict(r) for r in db.execute('SELECT m.person,m.status,COUNT(*) AS observations,'
            "SUM(m.role='anchor') AS anchors,SUM(m.role='auxiliary') AS auxiliary FROM body_members m "
            'JOIN body_observations b ON b.id=m.observation WHERE b.scan IN (SELECT value FROM json_each(?)) '
            'AND b.contract=? GROUP BY m.person,m.status ORDER BY m.person,m.status',(encoded(list(scans)),contract(config)))]
    return dict(result,assignments=dict(counts),reasons=dict(reasons),identities=identities)


def export_members(db,config,person,directory,*,policies=None):
    from .identity_report import _image
    from .person_audit import _draw
    from .alignment_audit import _png
    if not available(db):return []
    from . import admission_report, admission_panel
    policies = policies or admission_report.Policies(db,config,person['scope'],person['namespace'])
    invalid=db.execute('SELECT 1 FROM body_members m JOIN body_observations b ON b.id=m.observation '
        'JOIN scans s ON s.id=b.scan WHERE m.person=? AND (s.scope<>? OR s.namespace<>?) LIMIT 1',
        (person['id'],person['scope'],person['namespace'])).fetchone()
    if invalid:raise ValueError('Identity has incompatible retained person evidence.')
    samples=[]
    rows=db.execute('SELECT b.*,m.role,m.status AS assignment_status,m.policy,m.proof,s.scene,s.status AS scan_status '
        "FROM body_observations b JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan "
        "WHERE m.person=? ORDER BY s.scene,b.cue,b.id",(person['id'],))
    for index,row in enumerate(rows,1):
        scene=row['scene'];cue=row['cue']
        if not scene.isascii() or not scene.isdecimal() or int(scene)<=0 or type(cue) is not int or cue<0:
            raise ValueError('Invalid retained body scene/cue identifier.')
        geometry=json.loads(row['geometry']);quality=json.loads(row['quality'])
        name=f'scene-{int(scene)}_cue-{cue:06d}_person-{index:06d}.png'
        def annotate(rgb,path):
            bounds=geometry.get('bounds');points=geometry.get('keypoints')
            if not bounds or not points:return dict(status='missing_person_geometry',file=None)
            left,top,right,bottom=bounds
            if (right-left,bottom-top)!=rgb.size:return dict(status='incompatible_crop_geometry',file=None)
            shifted=[(x-left,y-top) if x>=0 and y>=0 else (-1,-1) for x,y in points]
            target=path.with_name(path.stem+'-annotated.png')
            with rgb.copy() as copy:
                drawn=_draw(copy,(0,0,rgb.width,rgb.height),shifted);_png(copy,target)
            return dict(status='exported',file=target.name,drawn_indices=drawn)
        image=_image(row['crop'],directory/name,annotate=annotate)
        admission=admission_report.person(quality,config,policies,row['contract'],independent=True)
        image['thresholds']=admission_panel.export(image,admission,directory,name,
            observation=row['id'],role=row['role'],rejection=row['rejected'])
        samples.append(dict(observation_id=row['id'],evidence_kind='person',linked_face=row['face'],
            scene_id=scene,scan_id=row['scan'],scan_status=row['scan_status'],cue_index=cue,seconds=row['seconds'],
            contract=row['contract'],contract_matches_current=row['contract']==contract(config),
            quality=quality,geometry=geometry,retained_rejection=row['rejected'],image=image,admission=admission,
            assignment=dict(role=row['role'],status=row['assignment_status']),decision=json.loads(row['proof']),
            anchor_pixel_gate=pixel_gate(row['quality'],config.data['sapiensid']['identities']),
            policy_matches_current=row['policy']==policy(config),
            current_anchor_eligible=(row['assignment_status']=='accepted' and row['role']=='anchor' and
                row['policy']==policy(config) and row['contract']==contract(config) and row['rejected'] is None)))
    return samples
