"""Deletion criteria, hierarchy pruning, fresh checks and API compatibility."""
from copy import deepcopy
from collections import Counter
import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.cleanup.prune import prune_tags, PruneClient, REQUIRED_COUNTS


from tests.helpers.prune import PruneStash


def test_prunes_unused_leaves_and_recursively_empty_prefixed_categories():
    stash=PruneStash()
    root=stash.add("op:category:Root")
    category=stash.add("op:category:General",[root])
    leaf=stash.add("op:old",[category])
    outside=stash.add("human empty")
    stats=prune_tags(stash,report=lambda _:None)
    assert stash.deleted==[leaf,category,root]
    assert set(stash.tags)=={outside} and stats["tags_pruned"]==3


@pytest.mark.parametrize("counter",sorted(REQUIRED_COUNTS-{"child_count"}|{"studio_count","group_count","movie_count"}))
def test_usage_in_any_supported_object_preserves_tag(counter):
    stash=PruneStash();identifier=stash.add("op:used",**{counter:1})
    assert prune_tags(stash,report=lambda _:None)["tags_pruned"]==0
    assert identifier in stash.tags and not stash.deleted


def test_parent_with_a_nonmatching_child_is_preserved():
    stash=PruneStash();parent=stash.add("op:parent");stash.add("manual child",[parent])
    assert prune_tags(stash,report=lambda _:None)["tags_pruned"]==0


def test_global_blacklist_prunes_only_exact_unused_definitions_and_reports_remaining_usage():
    stash=PruneStash()
    leaf=stash.add('op:blocked')
    used=stash.add('ai:blocked',image_count=1)
    parent=stash.add('wd:blocked');stash.add('child',[parent])
    outside=stash.add('op:unrelated_empty')
    stats=prune_tags(stash,names={'op:blocked','ai:blocked','wd:blocked'},report=lambda _:None)
    assert stash.deleted==[leaf]
    assert {used,parent,outside} <= set(stash.tags)
    assert stats['excluded_tags_still_used_or_parent']==2


def test_prefix_scope_and_shared_parent_are_respected():
    stash=PruneStash();parent=stash.add("op:parent")
    empty=stash.add("op:empty",[parent]);used=stash.add("op:used",[parent],scene_count=1)
    stash.add("wd:empty");stash.add("opportunistic:empty")
    assert prune_tags(stash,["op:"],report=lambda _:None)["tags_pruned"]==1
    assert stash.deleted==[empty] and parent in stash.tags and used in stash.tags


@pytest.mark.parametrize("change",["new_usage","new_child","renamed"])
def test_refresh_prevents_deleting_a_tag_that_changed_after_catalog(change):
    stash=PruneStash();identifier=stash.add("op:empty")
    def mutate(stash,identifier):
        if change=="new_usage":stash.tags[identifier]["scene_marker_count"]=1
        elif change=="new_child":stash.add("manual child",[identifier])
        else:stash.tags[identifier]["name"]="manual renamed"
        stash.before_refresh=None
    stash.before_refresh=mutate
    assert prune_tags(stash,report=lambda _:None)["tags_retained_after_refresh"]==1
    assert identifier in stash.tags and not stash.deleted


def test_dry_run_simulates_full_branch_without_mutation():
    stash=PruneStash(True)
    parent=stash.add("op:parent");stash.add("op:leaf",[parent])
    before=deepcopy(stash.tags)
    assert prune_tags(stash,report=lambda _:None)["tags_pruned"]==2
    assert stash.tags==before and not stash.deleted and not stash.calls["refresh"]


@pytest.mark.parametrize("missing",["scene_marker_count","studio_count","group_count"])
def test_missing_usage_capabilities_abort_before_any_delete(missing):
    stash=PruneStash();stash.counts.remove(missing);stash.add("op:empty")
    with pytest.raises(RuntimeError,match="all tag usage"):prune_tags(stash,report=lambda _:None)
    assert not stash.deleted


@pytest.mark.parametrize("value",[None,-1,False,"0"])
def test_unknown_counter_is_not_interpreted_as_zero(value):
    stash=PruneStash();stash.add("op:empty");stash.add("op:broken",image_count=value)
    with pytest.raises(ValueError,match="usage count"):prune_tags(stash,report=lambda _:None)
    assert not stash.deleted


def test_pagination_completes_before_deletion():
    stash=PruneStash()
    for i in range(1002):stash.add(f"op:empty_{i}")
    assert prune_tags(stash,report=lambda _:None)["tags_pruned"]==1002
    assert stash.calls["catalog"]==2


@pytest.mark.parametrize("prefixes",[[""],["op"],[":"],["op: "]])
def test_blank_or_ambiguous_prefix_is_rejected(prefixes):
    stash=PruneStash();stash.add("op:empty")
    with pytest.raises(ValueError,match="namespaces"):prune_tags(stash,prefixes,report=lambda _:None)
    assert not stash.calls


def test_standalone_cli_needs_no_models_or_hierarchy(monkeypatch,tmp_path):
    stash=PruneStash();stash.add("op:empty")
    monkeypatch.setattr(app,"DRY_RUN",False)
    monkeypatch.setattr(app.services,"create_stash_client",lambda _:stash)
    monkeypatch.setattr(app.services,"load_model_settings",lambda *args:pytest.fail("pruning loaded model configuration"))
    monkeypatch.setattr(app.services,"load_vocabulary",lambda *args:pytest.fail("pruning loaded vocabulary"))
    assert app.main(["--prune-empty-tags-only","--state-db",str(tmp_path/"state.sqlite3")])==0
    assert len(stash.deleted)==1

