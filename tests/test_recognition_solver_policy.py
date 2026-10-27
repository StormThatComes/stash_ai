from stash_ai.recognition.merge_goals import assess,objective
from stash_ai.recognition.merge_solver_metrics import magnitude
from stash_ai.recognition.merge_solver_search import BeamSearch,PREFIX
from tests.test_recognition_merge_best_effort import goal_sets


def test_all_subcomponents_contribute_after_largest_group_tie():
    goals=goal_sets([[str(i)] for i in range(8)])
    observations=dict.fromkeys(map(str,range(8)),'scene')
    first=assess(goals,observations,{},dict(zip(map(str,range(8)),'aaaabbcc')))
    second=assess(goals,observations,{},dict(zip(map(str,range(8)),'aaaabbbc')))
    assert first['groups'][0]['largest_complete_performers']==second['groups'][0]['largest_complete_performers']==4
    assert objective(second)>objective(first)
    assert second['groups'][0]['pair_consolidation_fraction']>first['groups'][0]['pair_consolidation_fraction']


def test_mutations_use_comparable_units():
    pixels=magnitude([dict(parameter='clustering.min_anchor_face_pixels',current=80,proposed=72)])
    support=magnitude([dict(parameter='recognition.min_merge_support',current=10,proposed=9)])
    assert pixels==support==.1


def test_same_partition_retains_distant_parameter_values():
    search=BeamSearch(['hybrid'],lambda *a:iter(()),500)
    search.observe('baseline',{},(1,0,0),'same')
    search.observe('near',{PREFIX+'face_floor':.78},(1,-1,-.02),'same',family='hybrid')
    search.observe('far',{PREFIX+'face_floor':.6},(1,-1,-.2),'same',family='hybrid')
    assert 'near' in search.beam and 'far' in search.beam


def test_stagnation_has_one_bounded_restart_and_keeps_best():
    def factory(parent,changes,family,mode):
        for i in range(200):
            if mode:yield 'face','restart_probe' if mode=='restart' else 'local_refinement',{'x':i}
            else:yield 'coordinate_probe',{'x':i}
    search=BeamSearch(['face'],factory,500)
    search.observe('baseline',{},(1,0,0),'baseline')
    phases=[]
    for i in range(200):
        try:family,phase,parent,strategy,changes=search.next()
        except StopIteration:break
        phases.append(phase);search.completed(phase)
        search.observe(f'trial-{i}',changes,(0,-1,-i),'loser',family=family,operator=strategy,elapsed=1)
    assert phases.count('restart')==8 and search.restarts==1
    assert 'baseline' in search.beam
    assert search.stop_reason=='stagnation_after_refinement'
