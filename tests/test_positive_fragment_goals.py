from stash_ai.recognition.positive_observations import assess
from stash_ai.recognition.sequence_validation import sequence_metrics


def test_provisional_body_requires_independently_recovered_owner():
    label = dict(set=0, scene='7', requested_observation_ids=['rear'], observation_ids=['rear'])
    goals = dict(positive_observations=[label], positive_reference_ids={0:['root']})
    for owner, passed in ((None, False), ('new', False), ('performer', True)):
        status = dict(admissible=True, fully_solved=True)
        membership = {'root':'performer', 'rear':owner}
        assess(goals,status,membership,membership)
        assert status['fully_solved'] == passed


def test_ordinary_switch_and_midshot_takeover():
    labels = {'a':'A','b':'B','c':'C','d':'C','e':'B','f':'B'}
    cues = dict(a=0,b=0,c=1,d=2,e=1,f=2)
    samples = {k:dict(slices=['entrant'] if person=='C' else []) for k,person in labels.items()}
    decisions = {k:dict(accepted_identity=v) for k,v in dict(a='1',b='2',c='1',d='3',e='2',f='2').items()}
    result = sequence_metrics(labels,samples,{k:('s',v) for k,v in cues.items()},decisions)
    assert result['entrant_takeovers']==result['scene_id_switches']==1


def test_positive_cli_label_freezes_current_cue_and_scores_linked_projections(tmp_path):
    from stash_ai.recognition.cli import parser
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.merge_goals import resolve
    from tests.test_recognition_observation_exclusions import linked_fixture

    cfg,faces,bodies,people = linked_fixture(tmp_path)
    with FaceDatabase(cfg.database) as db:
        performer = db.connection.execute('SELECT performer FROM people WHERE id=?',(people[0],)).fetchone()[0]
        args = parser().parse_args(['report-merge','--performer-set',performer,
                                    '--include-performer-observation',f'{performer}:{bodies[0][0]}'])
        goals = resolve(db.connection,cfg,[[performer]],observation_inclusions=args.include_performer_observation)
        label, = goals['positive_observations']
        assert label['cue']==0 and set(label['observation_ids'])=={faces[0],bodies[0][0]}
        assert not set(goals['positive_reference_ids'][0]).intersection(label['observation_ids'])


def test_reference_abstention_cannot_certify_positive():
    label = dict(set=0,scene='7',requested_observation_ids=['q'],observation_ids=['q'])
    goals = dict(positive_observations=[label], positive_reference_ids={0:['a','b']})
    for second in ('B',None):
        membership = dict(q='A',a='A',b=second)
        status = {}
        assess(goals,status,membership,membership)
        record, = status['positive_observations']
        assert not record['satisfied']
        assert record['lost_reference_units']==int(second is None)


def test_one_goal_per_track_counts_swaps_and_missing_runs():
    labels = [dict(set=0,scene='7',cue=i,track='fragment',requested_observation_ids=[str(i)],
                   observation_ids=[str(i)]) for i in range(5)]
    goals = dict(positive_observations=labels,positive_reference_ids={0:['ref']})
    membership = {'ref':'A','0':'A','1':None,'2':None,'3':'B','4':'A'}
    status = {}
    assess(goals,status,membership,membership)
    record, = status['positive_observations']
    assert record['coverage']==.4 and record['longest_unmatched_run']==3
    assert record['owner_switches']==2 and record['wrong_owner_samples']==1


def test_turn_routes_report_gates_that_were_used():
    from stash_ai.recognition.merge_solver_tighten import route_parameters
    for reason in ('bracketed_scene_core','bracketed_body_turn','corroborated_rear_view'):
        used=route_parameters([dict(status='accepted',reason=reason,evidence_kind='person',
                                     measurements=dict(track_floor=.75,turn_discount=.06))])
        assert used['sapiensid.identities.turn_similarity_discount']==1
        assert used['sapiensid.identities.track_similarity_threshold']==1


def test_positive_goal_uses_body_only_reference_quorum_and_manual_pair(tmp_path):
    from tests.test_recognition_observation_exclusions import linked_fixture
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.merge_solver_diagnostics import latest_decisions
    import json
    cfg, faces, bodies, _ = linked_fixture(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        body = bodies[0][0]
        cols = [r[1] for r in db.execute('PRAGMA table_info(body_observations)')]
        fields = {'id':"'corrected-body'", 'face':'NULL'}
        db.execute('INSERT INTO body_observations SELECT '+','.join(fields.get(c,c) for c in cols)+
                   ' FROM body_observations WHERE id=?',(body,))
        goals = resolve(db,cfg,[['900']],policy='identity-physical',physical_pairs=[faces[0]+':corrected-body'],
                        observation_inclusions=[('900',faces[0])])
        assert set(goals['positive_observations'][0]['observation_ids'])=={faces[0],'corrected-body'}
        predicted = {oid:'performer' for oid in [*bodies[0][1:],'corrected-body']}
        status = {}
        assess(goals,status,predicted,predicted)
        assert status['positive_observations'][0]['satisfied']
        db.execute('UPDATE body_members SET proof=? WHERE observation=?',(json.dumps(dict(
            reason='fragment_body_attachment',fragment='track',source_tracks=[bodies[0]])),body))
        goal = resolve(db,cfg,[['900']],track_inclusions=[('900',body)])
        assert goal['positive_observations'][0]['track']=='track'
        row, = latest_decisions(db,{body})
        assert row['source_tracks']==[bodies[0]] and set(row['witnesses'])==set(bodies[0])


def test_physical_copresence_never_suggests_looser_matching(tmp_path):
    from .test_recognition_body_identity import configuration
    from stash_ai.recognition.positive_observations import proposals
    cfg=configuration(tmp_path)
    goals={'positive_observations':[dict(set=0,observation_ids=['q'])]}
    row=dict(observation='q',reason='body_copresence_conflict',
             label_probe=dict(stage='body_similarity',expected='A',score=.77,runner=.4))
    assert proposals(cfg,goals,[row],{})==[]


def test_label_topology_breaks_equal_coverage_ties():
    from stash_ai.recognition.merge_goals import objective
    group=dict(assigned_fraction=1,complete_fraction=1,largest_complete_performers=1,consolidation_fraction=1,fragments=1)
    labels=[dict(set=0,scene='7',cue=i,track='track',requested_observation_ids=[str(i)],observation_ids=[str(i)]) for i in range(5)]
    def score(sequence):
        predicted=dict(ref='A',**dict(zip(map(str,range(5)),sequence)))
        result=dict(groups=[group],admissible=True,fully_solved=True,disjoint=True,cross_set_observation_pairs=0)
        assess(dict(positive_observations=labels,positive_reference_ids={0:['ref']}),result,predicted,predicted)
        return objective(result)
    assert score('AAABB')>score('ABABA')
