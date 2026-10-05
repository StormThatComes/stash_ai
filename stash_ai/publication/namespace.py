"""Stash output namespace metadata, separate from inference model identity."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputNamespace:
    key: str
    title: str
    prefix: str
    owner_tag: str
    title_prefix: str
    hierarchy_description: str
