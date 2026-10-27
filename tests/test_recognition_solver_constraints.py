from types import SimpleNamespace
from stash_ai.recognition.config import defaults
from stash_ai.recognition.merge_solver_constraints import joint_separation,crossovers,coupled_refinement,PREFIX
from stash_ai.recognition.merge_report_policy import patched
from tests.test_recognition_merge_solver_exploration import auxiliary_goals


def test_separation_uses_different_gates_for_different_rivals():
    config=SimpleNamespace(data=defaults())
    face,body=PREFIX+'face_floor',PREFIX+'person_threshold'
    config.data['sapiensid']['matching'].update(face_floor=.4,person_threshold=.4)
    positives=[{face:.8,body:.8}];negatives=[{face:.9,body:.5},{face:.5,body:.9}]
    result=joint_separation(config,positives,negatives)
    assert not result['unresolved']
    patch=result['proposals'][0]
    assert all(patch[k]<=v for k,v in positives[0].items())
    assert all(any(patch[k]>v for k,v in n.items()) for n in negatives)
    assert patch[face]>.5 and patch[body]>.5


def test_conflicting_witness_returns_explicit_partial_hint():
    config=SimpleNamespace(data=defaults());key=PREFIX+'face_floor'
    result=joint_separation(config,[{key:.5}],[{key:.6}])
    assert result['unresolved']==[0]
    assert result['proposals'][0][key]<=.5


def test_crossover_preserves_both_conflict_resolutions():
    result=list(crossovers({'a':1,'shared':2},[{'b':3,'shared':4}]))
    assert {'a':1,'b':3,'shared':2} in result
    assert {'a':1,'b':3,'shared':4} in result
    assert list(crossovers({'a':1,'b':2},[{'a':3,'b':4}]))


def test_joint_refinement_has_real_two_gate_mutations(tmp_path):
    config,_=auxiliary_goals(tmp_path)
    changes={PREFIX+'face_floor':.7,PREFIX+'person_threshold':.8}
    result=list(coupled_refinement(config,changes,{}))
    assert result and len(result)<=32
    assert all(sum(p[k]!=changes[k] for k in changes)==2 for p in result)
    for patch in result:patched(config,patch)
