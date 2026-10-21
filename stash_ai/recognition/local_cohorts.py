"""Freeze scene appearance priorities before creating any identity labels.

Only bounded raw exemplars live in Python; membership/order metadata lives in
the private RAM replay. Cue times still decide independent support and physical
co-presence. This priority is an appearance ordering, never a new timeline.
"""
from .clustering import vector
from .pose_policy import is_anchor,visibility_sql
from .reference_usage import ReferenceUsage
from .progress import tracking
from .view_prototypes import _Cluster,coverage_threshold


def ordering(db,alias='o'):
    if db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='local_core_order'").fetchone():
        return f'COALESCE((SELECT position FROM local_core_order WHERE observation={alias}.id),{alias}.cue)'
    return alias+'.cue'


def prepare(db,selected,config):
    if not config.data['clustering']['anchor_view_matching']:
        return
    import numpy as np
    grouping=config.data['clustering'];minimum=config.data['recognition']['min_identity_support']
    db.execute('CREATE TEMP TABLE local_core_order(observation TEXT PRIMARY KEY,position INTEGER)')
    db.execute('CREATE TEMP TABLE local_appearance_rows(observation TEXT PRIMARY KEY,cohort INTEGER)')
    usage=ReferenceUsage(grouping['max_reference_bytes'])
    with tracking(selected,'Freeze local appearance cohorts',unit='scene') as scenes:
        for item in scenes:
            usage.begin('local_appearance_cohorts',reset=True);slots=[];anchors=[]
            rows=db.execute('SELECT id,scan,cue,seconds,geometry,quality,dimensions,embedding FROM observations '
                'WHERE scan=? AND rejected IS NULL AND '+visibility_sql('observations',grouping.get('min_auxiliary_visible_face_fraction'))+' ORDER BY embedding,geometry,cue,id',(item['scan'],))
            for row in rows:
                sample=vector(row)
                slot=next((i for i,s in enumerate(slots) if s.accepts_pose(row['geometry']) and
                           s.accepts(sample,coverage_threshold(grouping))),None)
                if slot is None:
                    if len(slots)>=grouping['max_people_per_lookup']:
                        raise ValueError('Local appearance cohorts exceed configured person budget.')
                    slot=len(slots);slots.append(_Cluster('other',row['id']));anchors.append(set())
                cohort=slots[slot];before=len(cohort.examples);cohort.add(row,sample)
                usage.add_counts((len(cohort.examples)-before)*sample.nbytes,len(cohort.examples)-before,
                                 kind='Local appearance witnesses')
                if is_anchor(row['quality'],grouping):anchors[slot].add(row['seconds'])
                db.execute('INSERT INTO local_appearance_rows VALUES (?,?)',(row['id'],slot))
            if slots:
                centres=[s.medoid()[0] for s in slots]
                centre=np.mean(centres,axis=0,dtype='float64')
                remaining=set(range(len(slots)));placed=[]
                while remaining:
                    def priority(i):
                        connected=any(float(np.dot(centres[i],centres[j]))>=config.data['recognition']['match_threshold']
                                      for j in placed)
                        return (connected,len(anchors[i])>=minimum,max(slots[i].qualities,default=0.),
                                -float(np.dot(centres[i],centre)),tuple(float(x) for x in centres[i]),slots[i].pose)
                    winner=max(remaining,key=priority)
                    db.execute('INSERT INTO local_core_order SELECT observation,? FROM local_appearance_rows WHERE cohort=?',
                               (len(placed),winner))
                    placed.append(winner);remaining.remove(winner)
            db.execute('DELETE FROM local_appearance_rows')
    db.execute('DROP TABLE local_appearance_rows')
