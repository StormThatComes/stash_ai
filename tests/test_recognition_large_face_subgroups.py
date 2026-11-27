from stash_ai.recognition.frozen_subgroups import Edges


def test_large_component_can_recover_a_supported_triple_with_a_clear_boundary():
    graph = Edges(2**20)
    for i in range(129):
        graph.add(i,i+1,.81)
    for a,b in ((1,2),(2,3),(1,3)):
        graph.add(a,b,.99)
    assert (1,2,3) in graph.proposals(list(range(130)),.8,.05)
