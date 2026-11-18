"""Reversible raw-evidence review inside the disposable recognition replay."""
import json
import numpy as np
from .database import encoded
from .clustering import vector
from .independent_evidence import count
from .pose_policy import is_anchor


def overlap(a,b):
    a,b=np.asarray(a[:4],float),np.asarray(b[:4],float)
    area=np.maximum(0,np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])).prod()
    return area/max(1.,min(a[2:].prod(),b[2:].prod()))


def review(rows, grouping):
    """Independent clean witnesses on both sides may demote, never reassign."""
    maximum=grouping.get('anchor_review_max_cues',3)
    minimum=max(3,grouping.get('view_min_support',3))
    threshold=grouping['view_cohesion_threshold']
    margin=grouping['view_ambiguity_margin']
    by_cue={}
    for row in rows:by_cue.setdefault(row['cue'],[]).append(row)
    proof={}
    for row in rows:
        q=json.loads(row['quality']); shot=q.get('view_shot')
        if shot is None or not is_anchor(q,grouping):continue
        geometry=json.loads(row['geometry']); witnesses=[]
        for step in (-1,1):
            previous=row
            for cue in range(row['cue']+step,row['cue']+step*(maximum+1),step):
                candidates=by_cue.get(cue,())
                if not candidates:break
                ranked=sorted(((overlap(json.loads(previous['geometry']),json.loads(r['geometry'])),r['id'],r)
                               for r in candidates),reverse=True)
                score,_,candidate=ranked[0]
                if score<.5 or len(ranked)>1 and score-ranked[1][0]<.15:break
                cq=json.loads(candidate['quality'])
                later=cq if step==1 else json.loads(previous['quality'])
                if cq.get('view_shot')!=shot or not later.get('transition_verified'):break
                if is_anchor(cq,grouping):witnesses.append(candidate)
                previous=candidate
        if (count(witnesses)<minimum or not any(r['cue']<row['cue'] for r in witnesses)
                or not any(r['cue']>row['cue'] for r in witnesses)):continue
        values=[vector(r) for r in witnesses]
        if min(float(a@b) for i,a in enumerate(values) for b in values[i+1:])<threshold:continue
        # Every independent witness contradicts the suspect. A sole weak pair
        # cannot remove an otherwise supported, legitimate perspective.
        highest=max(float(vector(row)@v) for v in values)
        if highest>=grouping['cohesion_threshold']-margin:continue
        proof[row['id']]=dict(kind='temporal_anchor_contradiction',witness_ids=[r['id'] for r in witnesses],
            maximum_similarity=highest,reference_floor=threshold,query_only=True)
    return proof


def apply(db, selected, config):
    grouping=config.data['clustering']
    if not grouping.get('revalidate_face_anchors') and not grouping.get('near_duplicate_pixel_rms'):
        return
    for item in selected:
        rows=[dict(r) for r in db.execute('SELECT id,scan,cue,seconds,geometry,quality,dimensions,embedding '
            'FROM observations WHERE scan=? AND rejected IS NULL AND embedding IS NOT NULL ORDER BY cue,id',(item['scan'],))]
        if grouping.get('near_duplicate_pixel_rms'):
            collapse(db,rows,grouping)
        if grouping.get('revalidate_face_anchors'):
            for oid,proof in review(rows,grouping).items():
                db.execute("UPDATE observations SET quality=json_set(quality,'$.query_only',json('true'),"
                    "'$.anchor_review',json(?)) WHERE id=?",(encoded(proof),oid))


def collapse(db, rows, grouping):
    """Scene-local, non-transitive thumbnail proof; never an identity merge."""
    from .pixel_storage import read
    representatives=[]
    rms=grouping['near_duplicate_pixel_rms']
    similarity=grouping.get('near_duplicate_similarity',.995)
    for row in rows:
        q=json.loads(row['quality']); raw=read(db,q)
        if raw is None or len(raw)!=3072 or q.get('view_shot') is None:continue
        pixels=np.frombuffer(raw,np.uint8).astype(float)
        if pixels.std()<12:continue
        geometry=json.loads(row['geometry']); value=vector(row)
        for representative,other,feature,old,shape in representatives:
            if (other.get('view_shot')!=q['view_shot'] or representative['cue']==row['cue']
                    or overlap(geometry,shape)<.7 or float(feature@value)<similarity):continue
            aspect=q.get('evidence_aspect'); previous=other.get('evidence_aspect')
            if not aspect or not previous or abs(aspect/previous-1)>.02:continue
            if np.sqrt(np.mean((pixels-old)**2))>rms:continue
            from .config import digest
            unit=digest(['independent-face',row['scan'],representative['id']])
            for target,record in ((row,q),(representative,other)):
                record['evidence_independence_unit']=unit
                target['quality']=encoded(record)
                db.execute('UPDATE observations SET quality=? WHERE id=?',(target['quality'],target['id']))
            break
        else:
            representatives.append((row,q,value,pixels,geometry))


def publish_proofs(db):
    """Keep the review explanation in derived decisions, without editing raw data."""
    for oid,proof in db.execute("SELECT id,json_extract(quality,'$.anchor_review') FROM observations "
                               "WHERE json_type(quality,'$.anchor_review')='object'"):
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.anchor_review',json(?)) "
                   "WHERE json_extract(payload,'$.observation')=?",(proof,oid))
