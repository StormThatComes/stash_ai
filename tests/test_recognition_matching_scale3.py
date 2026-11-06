import random
from stash_ai.recognition.appearance_index import AppearanceIndex
from stash_ai.recognition.cue_presence import CueIndex


def test_hamming_index_matches_first_linear_representative():
    rng=random.Random(47);index=AppearanceIndex();old=[]
    values=[rng.getrandbits(64) for _ in range(300)]
    for value in values:
        for radius in (0,1,4,16,32,64,65):
            expected=next((p for v,p in old if (v^value).bit_count()<radius),None)
            assert index.first(value,radius)==expected
        index.add(value,len(old));old.append((value,len(old)))
    for value in values[::7]:
        assert index.first(value^3,4)==next(p for v,p in old if (v^(value^3)).bit_count()<4)


def test_cue_spill_keeps_exact_overlap_and_closes():
    pairs=[('a','b'),('a','c'),('b','c'),('a','missing')]
    indexes=[CueIndex(100000),CueIndex(0)]
    try:
        for index in indexes:
            for row in [('a','scan',1),('b','scan',1),('c','other',1),('a','scan',2)]:index.add(*row)
            assert index.overlap(pairs)==[('a','b')]
        assert indexes[0].db is None and indexes[1].db is not None
    finally:
        for index in indexes:index.close()
    assert all(i.db is None for i in indexes)
