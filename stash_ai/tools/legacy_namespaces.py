"""Retired per-model publishing metadata used only by explicit migrations."""
from stash_ai.publication.namespace import OutputNamespace


NAMESPACES = {source.key: source for source in (
    OutputNamespace('joytag', 'JoyTag', 'jt:', 'jt:marker:joytag', '[JoyTag] ', '[JoyTag hierarchy v1]'),
    OutputNamespace('wd', 'WD EVA02-Large v3', 'wd:', 'wd:marker:wd', '[WD] ', '[WD hierarchy v1]'),
    OutputNamespace('dtq', 'DanbooruTagQuery L/16', 'dtq:', 'dtq:marker:dtq', '[DTQ] ', '[DTQ hierarchy v1]'),
    OutputNamespace('oppai', 'OppaiOracle V1.1', 'op:', 'op:marker:oppai', '[OppaiOracle] ', '[OppaiOracle hierarchy v1]'),
)}
