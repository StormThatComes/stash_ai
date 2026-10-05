"""Publishing metadata is independent of inference model identity."""
from dataclasses import fields
import inspect
from stash_ai.models import ModelSource, SOURCES
from stash_ai.evidence.labels import source_codes
from stash_ai.publication.markers import owned_marker
from stash_ai.tools.legacy_namespaces import NAMESPACES


def test_historical_ownership_requires_explicit_namespace():
    assert inspect.signature(owned_marker).parameters['source'].default is inspect.Parameter.empty
    assert set(NAMESPACES) == set(SOURCES)
    assert not {'prefix', 'owner_tag', 'title_prefix', 'hierarchy_description'} & {field.name for field in fields(ModelSource)}
    assert source_codes(SOURCES) == ['jt', 'wd', 'dtq', 'op']
