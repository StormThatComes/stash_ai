"""Explicit injectable services used by command and run orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from stash_ai import config
from stash_ai.backends import loader, vocabulary
from stash_ai.stash_api.client import StashClient
from stash_ai.stash_api.tags import TagResolver
from stash_ai.runtime.settings import RuntimeSettings, validate_config
from stash_ai.runtime.summary import print_hierarchy_summary
from stash_ai.stash_api.transport import safe_error
from stash_ai.vtt import format_time


@dataclass
class RuntimeServices:
    """Factories are bound to one runtime; tests can replace individual services."""

    create_stash_client: Callable
    load_model_settings: Callable
    validate_config: Callable
    load_tagger: Callable = loader.load_tagger
    load_vocabulary: Callable = vocabulary.load_vocabulary
    ensure_files: Callable = vocabulary.ensure_files
    weight_files: Callable = vocabulary.weight_files
    load_unique_yaml: Callable = config.load_unique_yaml
    load_exclusion_policy: Callable = config.load_exclusion_policy
    load_tag_hierarchy: Callable = config.load_tag_hierarchy
    metadata_hierarchy: Callable = config.metadata_hierarchy
    vocabulary_digest: Callable = config.vocabulary_digest
    StashClient: Callable = StashClient
    TagResolver: Callable = TagResolver
    print_hierarchy_summary: Callable = print_hierarchy_summary
    safe_error: Callable = safe_error
    format_time: Callable = format_time

    @classmethod
    def bind(cls, runtime):
        return cls(
            create_stash_client=lambda dry_run: StashClient(
                dry_run, url=runtime.STASH_URL, api_key=runtime.STASH_API_KEY,
                request_timeout=runtime.REQUEST_TIMEOUT, http_retries=runtime.HTTP_RETRIES,
                scene_page_size=runtime.SCENE_PAGE_SIZE, create_markers=runtime.CREATE_MARKERS),
            load_model_settings=lambda *args, **kwargs: config.load_model_settings(
                *args, **kwargs,
                joytag_scene_threshold=runtime.JOYTAG_DEFAULT_SCENE_THRESHOLD,
                default_marker_threshold=runtime.DEFAULT_MARKER_THRESHOLD,
                joytag_batch_size=runtime.JOYTAG_DEFAULT_BATCH_SIZE),
            validate_config=partial(validate_config, runtime),
        )


@dataclass
class RuntimeContext(RuntimeSettings):
    """Settings plus explicit factories for one command invocation."""

    services: RuntimeServices | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.services is None:
            self.services = RuntimeServices.bind(self)
