"""Freeze scene appearance priorities before creating any identity labels.

Only bounded raw exemplars live in Python; membership/order metadata lives in
the private RAM replay. Cue times still decide independent support and physical
co-presence. This priority is an appearance ordering, never a new timeline.
"""
from .clustering import vector
from .pose_policy import is_anchor,visibility_sql
from .reference_usage import ReferenceUsage
from .progress import tracking
from .view_prototypes import _Cluster,coverage_threshold,best_cohort


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
    with tracking(selected,'Freeze local appearance cohorts',unit='scene',position=0) as scenes:
        for item in scenes:
            usage.begin('local_appearance_cohorts',reset=True);slots=[];anchors=[]
            condition = ('WHERE scan=? AND rejected IS NULL AND ' +
                         visibility_sql('observations',grouping.get('min_auxiliary_visible_face_fraction')))
            rows=db.execute('SELECT id,scan,cue,seconds,geometry,quality,dimensions,embedding FROM observations ' +
                condition + ' ORDER BY embedding,geometry,cue,id',(item['scan'],))
            with tracking(rows, 'Group local face observations', unit='observation',
                          total=lambda: db.execute('SELECT COUNT(*) FROM observations ' + condition,
                                                   (item['scan'],)).fetchone()[0]) as tracked:
                for row in tracked:
                    sample=vector(row)
                    slot=best_cohort(slots,row,sample,coverage_threshold(grouping))
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
                # Reassess only the appearance priority against frozen witnesses.
                # Moving a priority cannot train a cohort or create new support.
                anchors=[set() for _ in slots]
                reassessment=db.execute('SELECT o.*,a.cohort FROM observations o JOIN local_appearance_rows a '
                                      'ON a.observation=o.id ORDER BY o.id')
                with tracking(reassessment,'Reassess local appearance priorities',unit='observation') as tracked:
                    for row in tracked:
                        slot=best_cohort(slots,row,vector(row),coverage_threshold(grouping))
                        slot=row['cohort'] if slot is None else slot
                        db.execute('UPDATE local_appearance_rows SET cohort=? WHERE observation=?',(slot,row['id']))
                        if is_anchor(row['quality'],grouping):anchors[slot].add(row['seconds'])
                centres=[s.medoid()[0] for s in slots]
                centre=np.mean(centres,axis=0,dtype='float64')
                remaining=set(range(len(slots)));placed=[]
                connected=[False]*len(slots)
                priorities=[(len(anchors[i])>=minimum,max(slots[i].qualities,default=0.),
                             -float(np.dot(c,centre)),tuple(float(x) for x in c),slots[i].pose)
                            for i,c in enumerate(centres)]
                while remaining:
                    winner=max(remaining,key=lambda i:(connected[i],)+priorities[i])
                    db.execute('INSERT INTO local_core_order SELECT observation,? FROM local_appearance_rows WHERE cohort=?',
                               (len(placed),winner))
                    placed.append(winner);remaining.remove(winner)
                    for i in remaining:
                        if not connected[i] and float(np.dot(centres[i],centres[winner]))>=config.data['recognition']['match_threshold']:
                            connected[i]=True
            db.execute('DELETE FROM local_appearance_rows')
    db.execute('DROP TABLE local_appearance_rows')
