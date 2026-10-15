"""Coordinator-owned, invocation-local immutable recognition work."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType

from .config import digest

_policies = ContextVar('recognition_policies', default=None)
_references = ContextVar('recognition_portrait_references', default=None)


@contextmanager
def policy_scope():
    token = _policies.set({})
    try:
        yield
    finally:
        _policies.reset(token)


def policy_digest(config, key, build):
    cache = _policies.get()
    # Injected mutable configurations must never acquire stale identities.
    if cache is None or not isinstance(config.data, MappingProxyType):
        return digest(build())
    identity = id(config), key
    if identity not in cache:
        cache[identity] = config, digest(build())
    return cache[identity][1]


@contextmanager
def portrait_reference_scope():
    """Only the metadata phase borrows this snapshot: membership is finalized."""
    token = _references.set({})
    try:
        yield
    finally:
        _references.reset(token)


def portrait_references(database, scope, namespace, config, load):
    from .clustering import reference_context
    cache = _references.get()
    key = database, scope, namespace, id(config)
    if cache is not None and key in cache:
        return cache[key][1:]
    groups = {group.identifier: group for group in load()}
    context = reference_context(groups.values())
    if cache is not None:
        cache[key] = config, groups, context
    return groups, context
