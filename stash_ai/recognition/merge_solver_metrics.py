"""Comparable mutation sizes and parameter-space diversity."""
from .merge_report_policy import parameter


def scale(key, current):
    if key.endswith(('face_pixels','person_pixels')):return max(32.,float(current or 0))
    if key.endswith('sharpness'):return max(10.,float(current or 0))
    if key.endswith('support'):return max(1.,float(current or 0))
    if key.endswith(('keypoints','appearance_distance')):return max(1.,float(current or 0))
    return 1.


def magnitude(changes):
    return sum(abs(float(c['proposed'])-float(c['current']))/scale(c['parameter'],c['current']) for c in changes)


def signature(changes, config=None):
    def value(key,v):
        current=parameter(config.data,key) if config else v
        return round(float(v)/scale(key,current)/.02) if type(v) is not bool else v
    return tuple((k,value(k,v)) for k,v in sorted(changes.items()))
