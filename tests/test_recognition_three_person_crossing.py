from stash_ai.recognition.crossing_tracks import crossings


def test_three_person_crossing_requires_unique_complete_assignment():
    frames={cue:[dict(id=f'{cue}-{lane}',cue=cue,lane=lane,face_person=str(lane) if cue!=1 else None)
                 for lane in range(3)] for cue in range(3)}
    roots={r['id']:str(r['lane']) for cue in (0,2) for r in frames[cue]}
    decisions={r['id']:dict(person=r['face_person']) for rows in frames.values() for r in rows}
    score=lambda a,b:.99 if a['lane']==b['lane'] else .6
    transition=lambda a,b:score(a,b) if score(a,b)>.8 else None
    claims=crossings(frames,roots,decisions,transition,.1,raw=score,max_people=3)
    assert {oid:proof[0] for oid,proof in claims.items()}=={f'1-{i}':str(i) for i in range(3)}
    assert not crossings(frames,roots,decisions,lambda a,b:.99,.1,raw=lambda a,b:.99,max_people=3)


def test_four_person_contact_with_two_visible_people_and_an_entrant():
    frames={cue:[dict(id=f'{cue}-{lane}',cue=cue,lane=lane,face_person=str(lane) if cue!=1 else None)
                 for lane in range(4) if cue!=1 or lane in (0,2)] for cue in range(3)}
    roots={r['id']:str(r['lane']) for cue in (0,2) for r in frames[cue]}
    decisions={r['id']:dict(person=r['face_person']) for rows in frames.values() for r in rows}
    score=lambda a,b:.99 if a['lane']==b['lane'] else .5
    transition=lambda a,b:score(a,b) if score(a,b)>.8 else None
    claims=crossings(frames,roots,decisions,transition,.1,raw=score,
                     occluded={1},visible_single=lambda r:True,max_people=4)
    assert {oid:proof[0] for oid,proof in claims.items()}=={'1-0':'0','1-2':'2'}
    frames[1][0]['face_person']='new'
    decisions['1-0']['person']='new'
    assert not crossings(frames,roots,decisions,transition,.1,raw=score,
                         occluded={1},visible_single=lambda r:True,max_people=4)
