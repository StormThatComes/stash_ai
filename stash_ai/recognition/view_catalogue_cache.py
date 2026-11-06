"""Portable cached Views retain their exact supporting source IDs and scenes."""
import base64
import numpy as np


def encode(views):
    result=[]
    for v in views:
        def vector(x):return base64.b64encode(np.asarray(x,dtype='<f4').tobytes()).decode('ascii')
        result.append(dict(person=v.group.identifier,samples=[vector(x) for x in v.group.samples],sample_ids=v.group.sample_ids,
            pose=v.pose,medoid=v.medoid,support=v.support,support_count=v.support_count,examples=[vector(x) for x in v.examples],
            authority=v.authority,promotion=v.promotion,source_scenes=v.source_scenes,scene_sources=v.scene_sources))
    return result


def decode(records):
    from .global_views import IdentityView
    from .clustering import Group
    from .paged_references import pack
    result=[]
    for row in records:
        def vectors(key):return pack(np.frombuffer(base64.b64decode(raw,validate=True),dtype='<f4') for raw in row[key])
        result.append(IdentityView(Group(row['person'],vectors('samples'),True,tuple(row['sample_ids'])),
            row['pose'],row['medoid'],tuple(row['support']),row['support_count'],vectors('examples'),row['authority'],
            row['promotion'],tuple(row['source_scenes']),tuple((s,tuple(ids)) for s,ids in row['scene_sources'])))
    return tuple(result)
