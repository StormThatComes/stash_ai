import json
import sqlite3
from contextlib import closing

import numpy as np

from stash_ai.recognition.perspective_pages import DiskGallery
from stash_ai.recognition.rear_gallery import Gallery,winner


def test_disk_banks_match_exact_radius_winner_and_runner_for_each_angle():
    settings = dict(match_threshold=.85,ambiguity_margin=.05,cohesion_threshold=.8,attach_min_support=2)
    banks = []
    for owner,kind,angle in [('A','back',0.),('B','side',.07),('C','side',2.)]:
        rows = [dict(id=f'{owner}{i}',scan=f's{owner}{i}',roots=['a','b'],perspective=kind,
                     quality=json.dumps(dict(quality=dict(pixel_digest=str(i)))),
                     _vector=np.array([np.cos(angle),np.sin(angle)],dtype='<f4')) for i in range(2)]
        banks.append((owner,rows[0]['scan'],rows))
    ram = Gallery(banks)
    query = np.array([1.,0.],dtype='<f4')
    with closing(sqlite3.connect(':memory:')) as db:
        disk = DiskGallery(db,banks,8*1024**2)
        for scan in ('other','sA0'):
            for kind in ('back','side'):
                expected = ram.rank(query,scan,settings,None,kind)
                actual = disk.rank(query,scan,settings,None,kind)
                assert [r[:4]+(r[5],) for r in actual]==[r[:4]+(r[5],) for r in expected]
                assert winner(actual,settings)==winner(expected,settings)==None
        for radius in (0.,.1,1.,2.1):
            assert disk.nearby(query,radius)==ram.nearby(query,radius)
        solo = Gallery([banks[1]])
        assert winner(solo.rank(query,'new',settings,None,'side'),settings)[2]=='B'
        assert winner(solo.rank(query,'new',settings,None,'back'),settings) is None
