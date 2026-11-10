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
