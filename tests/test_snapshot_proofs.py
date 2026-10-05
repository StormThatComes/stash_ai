"""Compact snapshots retain publication, cleanup, recovery and preview semantics."""
from copy import deepcopy
import json
import zlib

import pytest

from stash_ai.rethreshold import execution as rethreshold_execution
from stash_ai import db
from tests.helpers.runtime import runtime as app
from stash_ai.storage import recovery as recovery
import stash_ai.publication.recovery as recovery_recovery
from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.store as merge_store
from stash_ai.cleanup import state as cleanup
from stash_ai.cleanup import evidence as cleanup_evidence
from stash_ai.publication import merged as merge_sync
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import StateStore
from stash_ai.storage.codec import decode, encode
from stash_ai.storage.snapshots import FORMAT_KEY, FORMAT, compact, expand, decode as decode_snapshot
from tests.helpers.evidence import empty_policies, segment, span
from tests.helpers.stash import Stash
from tests.test_result_thresholds import config, outputs

pytestmark = pytest.mark.usefixtures('reset_temporal')


def shared_snapshot():
    proof = dict(intervals=[[float(i), float(i)+.75] for i in range(200)],
                 sample_times=[0., 1., 2.], hits=200,
                 runs=[[0, 200, 0., 200.]], run_group='original-input',
                 sources={'wd': dict(score=.9, intervals=[[0., 200.]],
                                     sample_times=[0., 1.], signatures=['original-signature'])})
    contributions = [span('wd', 0, 200, tag, extras=['extra']) for tag in ('first', 'second')]
    for item in contributions:
        item['extra_support'] = {'extra': proof}
    contributions[-1]['extra_support']['unique'] = dict(intervals=[[10., 11.]], sample_times=[10.])
    return dict(contributions=contributions, owned_ids=['manual-id'], cleanup_generation=4)


def plain(value):
    return json.loads(json.dumps(value))


def raw_snapshots(store):
    return store.connection.execute('SELECT * FROM snapshots ORDER BY scope,scene,kind').fetchall()


def test_codec_shares_only_repeated_proofs_and_preserves_borrowed_input():
    saved = shared_snapshot()
    original = deepcopy(saved)
    packed = compact(saved)
    assert saved == original
    assert packed[FORMAT_KEY] == FORMAT and len(packed['proofs']) == 1
    items = packed['snapshot']['contributions']
    assert items[0]['extra_support']['extra'] == items[1]['extra_support']['extra'] == 0
    assert isinstance(items[1]['extra_support']['unique'], dict)
    restored = decode_snapshot(encode(packed))
    assert restored == plain(saved)
    assert restored['contributions'][0]['extra_support']['extra'] is restored['contributions'][1]['extra_support']['extra']
    assert len(zlib.decompress(encode(packed))) < len(zlib.decompress(encode(saved)))
    # Reference numbers remain stable when callers insert support keys differently.
    saved['contributions'][-1]['extra_support'] = dict(reversed(list(saved['contributions'][-1]['extra_support'].items())))
    assert encode(compact(saved)) == encode(packed)


@pytest.mark.parametrize('value', [{}, {'contributions': []}])
def test_unshared_snapshots_always_use_the_versioned_envelope(value):
    packed = compact(value)
    assert packed[FORMAT_KEY] == FORMAT and packed['proofs'] == []
    assert decode_snapshot(encode(packed)) == value
    assert decode(encode(packed)) == packed  # Plain JSON never guesses a domain.


@pytest.mark.parametrize('value', [None, [], {'contributions': [None]},
    {'contributions': [{'extra_support': []}]},
    {'contributions': [{'extra_support': {'tag': None}}]}])
def test_invalid_snapshot_inputs_are_rejected(value):
    with pytest.raises(ValueError, match='compact snapshot'):
        compact(value)


def test_equal_independently_owned_proofs_remain_inline():
    saved = shared_snapshot()
    saved['contributions'][1]['extra_support']['extra'] = deepcopy(saved['contributions'][0]['extra_support']['extra'])
    packed = compact(saved)
    assert packed['proofs'] == []
    assert packed['snapshot'] == saved


@pytest.mark.parametrize('kind', ['markers', 'scene', 'nested_scene'])
@pytest.mark.parametrize('damage', ['thresholds', 'exclusions', 'source', 'settings', 'blocklist'])
def test_current_snapshots_reject_incomplete_build_options_on_write_and_read(kind, damage):
    from dataclasses import asdict
    from stash_ai.evidence.temporal.policy import build_options
    options = ({**build_options(app), 'core_exclusions': {s: [] for s in empty_policies()},
                'tag_exclusions': {s: [] for s in empty_policies()}} if kind == 'markers' else
               {'policy': asdict(app.TEMPORAL_POLICY), 'max_tags': 0, 'scene_thresholds': {},
                'scene_exclusions': {s: [] for s in empty_policies()}})
    threshold_field = 'marker_thresholds' if kind == 'markers' else 'scene_thresholds'
    exclusion_field = 'tag_exclusions' if kind == 'markers' else 'scene_exclusions'
    if damage == 'thresholds':
        del options[threshold_field]
    elif damage == 'exclusions':
        del options[exclusion_field]
    elif damage == 'source':
        del options[exclusion_field]['wd']
    elif damage == 'settings':
        if kind == 'markers':
            del options['settings']['MARKER_MIN_HITS']
        else:
            del options['max_tags']
    else:
        options['blocklist'] = []
    item = {'build_options': options}
    saved = {'contributions': [{'scene_evidence': [item]} if kind == 'nested_scene' else item]}
    before = deepcopy(saved)
    with pytest.raises(ValueError, match='frozen'):
        compact(saved)
    packed = {FORMAT_KEY: FORMAT, 'snapshot': saved, 'proofs': []}
    with pytest.raises(ValueError, match='frozen'):
        decode_snapshot(encode(packed))
    assert saved == before


@pytest.mark.parametrize('damage', ['format', 'bool_format', 'missing', 'unexpected', 'snapshot',
    'contributions', 'item', 'proofs', 'proof', 'support', 'negative', 'outside', 'bool_ref', 'null_ref', 'string_ref'])
def test_invalid_envelopes_or_references_are_rejected(damage):
    value = plain(compact(shared_snapshot()))
    if damage == 'format': value[FORMAT_KEY] = 999
    elif damage == 'bool_format': value[FORMAT_KEY] = True
    elif damage == 'missing': del value['proofs']
    elif damage == 'unexpected': value['unknown'] = True
    elif damage == 'snapshot': value['snapshot'] = []
    elif damage == 'contributions': value['snapshot']['contributions'] = {}
    elif damage == 'item': value['snapshot']['contributions'][0] = None
    elif damage == 'proofs': value['proofs'] = {}
    elif damage == 'proof': value['proofs'][0] = []
    elif damage == 'support': value['snapshot']['contributions'][0]['extra_support'] = []
    else:
        value['snapshot']['contributions'][0]['extra_support']['extra'] = {
            'negative': -1, 'outside': 1, 'bool_ref': True, 'null_ref': None, 'string_ref': '0'}[damage]
    before = deepcopy(value)
    with pytest.raises(ValueError, match='compact snapshot'):
        decode_snapshot(encode(value))
    with pytest.raises(ValueError, match='compact snapshot'):
        expand(value)
    assert value == before


@pytest.mark.parametrize('operation', ['prepare', 'publish'])
@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('cache_bytes', [0, 2**20])
def test_snapshot_storage_promotion_and_preview_are_transparent(tmp_path, monkeypatch, operation, dry_run, cache_bytes):
    path = tmp_path/'state.sqlite3'
    saved = shared_snapshot()
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        store.publish('1', {'markers': {'contributions': []}})
    source = path.read_bytes()
    with StateStore(path, 'scope', dry_run=dry_run, cache_bytes=cache_bytes) as store:
        assert store.connection.execute('PRAGMA user_version').fetchone() == (12,)
        store.report = RunReport(write=lambda _: None)
        raw_results = store.connection.execute('SELECT * FROM ' + store.table('results')).fetchall()
        getattr(store, operation)('1', {'markers': saved})
        assert store.connection.execute('PRAGMA user_version').fetchone() == (12,)
        column = 'pending' if operation == 'prepare' else 'published'
        blob = store.connection.execute('SELECT '+column+' FROM snapshots').fetchone()[0]
        assert json.loads(zlib.decompress(blob))[FORMAT_KEY] == FORMAT
        if operation == 'prepare':
            assert store.snapshot('1', 'markers')[1] == saved
            with monkeypatch.context() as patch:
                patch.setattr(store, 'pack', lambda *_: pytest.fail('Promotion encoded snapshot again'))
                store.commit_prepared('1', ['markers'])
            assert store.connection.execute('SELECT published FROM snapshots').fetchone()[0] == blob
        borrowed = store.snapshot('1', 'markers', borrow=True)[0]
        assert borrowed == saved
        detached = store.snapshot('1', 'markers')[0]
        detached['contributions'][0]['extra_support']['extra']['intervals'].clear()
        assert store.snapshot('1', 'markers')[0] == saved
        assert store.connection.execute('SELECT * FROM '+store.table('results')).fetchall() == raw_results
    if dry_run:
        assert path.read_bytes() == source
    else:
        before = path.read_bytes()
        with StateStore(path, 'scope', dry_run=True) as preview:
            assert preview.preview_source  # v9 stays a sparse read-only attachment.
            assert preview.connection.execute('SELECT COUNT(*) FROM main.snapshots').fetchone() == (0,)
            assert preview.snapshot('1', 'markers') == (saved, None)
        assert path.read_bytes() == before
        assert db.check_database(path, 'scope')['ok']


@pytest.mark.parametrize('operation', ['prepare', 'publish'])
@pytest.mark.parametrize('version', [5, 8])
def test_compact_write_and_schema_upgrade_roll_back_together(tmp_path, operation, version):
    with StateStore(tmp_path/'state.sqlite3', 'scope', cache_bytes=2**20) as store:
        store.put('1', 'wd', outputs())
        store.publish('1', {'markers': {'contributions': []}})
        store.connection.execute(f'PRAGMA user_version={version}')
        before = raw_snapshots(store)
        with pytest.raises(ValueError):
            getattr(store, operation)('1', {'markers': shared_snapshot(), 'scene': {'bad': float('nan')}})
        assert store.connection.execute('PRAGMA user_version').fetchone() == (version,)
        assert raw_snapshots(store) == before
        assert store.snapshot('1', 'markers') == ({'contributions': []}, None)


def test_rethreshold_cleanup_and_abandonment_never_downgrade_v9(tmp_path):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        store.publish('1', {'markers': shared_snapshot()})
        assert rethreshold_execution.run(app, store, ['1'], [config(.8)], RunReport(write=lambda _: None), markers=True) == {'scene_thresholds_updated': 1}
        assert store.connection.execute('PRAGMA user_version').fetchone() == (12,)
        cleanup.begin(store, 'policy', {'unused'}, set(), [])
        cleanup.finish(store)
        store.prepare('1', {'markers': shared_snapshot()})
        pending = store.connection.execute('SELECT pending FROM snapshots').fetchone()[0]
        recovery.abandon(store, lambda _: None)
        assert store.connection.execute('SELECT data FROM recovery_publications').fetchone()[0] == pending
        assert store.connection.execute('PRAGMA user_version').fetchone() == (12,)
        assert recovery.publications(store, '1', 'markers')[0]['contributions'] == shared_snapshot()['contributions']
    assert db.inspect_database(path, 'scope')['schema_version'] == 12
    assert db.check_database(path, 'scope')['ok']


def test_lazy_cleanup_masks_shared_proofs_and_recovery_without_rewriting_blobs(tmp_path):
    path = tmp_path/'state.sqlite3'
    saved = shared_snapshot()
    saved['cleanup_generation'] = 0
    with StateStore(path, 'scope', cache_bytes=2**20) as store:
        store.publish('1', {'markers': saved})
        store.prepare('1', {'markers': saved})
        recovery.abandon(store, lambda _: None)
        borrowed = store.snapshot('1', 'markers', borrow=True)[0]
        published = store.connection.execute('SELECT published FROM snapshots').fetchone()[0]
        archived = store.connection.execute('SELECT data FROM recovery_publications').fetchone()[0]
        cleanup.begin(store, 'policy', {'unique'}, set(), {'manual-id'})
        cleanup.finish(store)
        masked = store.snapshot('1', 'markers', borrow=True)[0]
        recovered = recovery.publications(store, '1', 'markers')[0]
        for value in (masked, recovered):
            assert value['cleanup_generation'] == 1 and not value['owned_ids']
            first, second = value['contributions']
            assert 'unique' not in second['extra_support']
            assert first['extra_support']['extra'] is second['extra_support']['extra']
        assert borrowed == saved  # Lazy masks never mutate earlier borrowed views.
        assert store.connection.execute('SELECT published FROM snapshots').fetchone()[0] == published
        assert store.connection.execute('SELECT data FROM recovery_publications').fetchone()[0] == archived
        store.publish('1', {'markers': masked})
        assert store.snapshot('1', 'markers')[0] == masked
    assert db.check_database(path, 'scope')['ok']


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery'])
def test_db_check_detects_corrupt_proof_references_without_repair(tmp_path, location):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', outputs())
        store.publish('1', {'markers': shared_snapshot()})
        store.prepare('1', {'markers': shared_snapshot()})
        if location == 'recovery':
            recovery.abandon(store, lambda _: None)
        damaged = compact(shared_snapshot())
        damaged['snapshot']['contributions'][0]['extra_support']['extra'] = 999
        with store.connection:
            if location == 'recovery':
                store.connection.execute('UPDATE recovery_publications SET data=?', (encode(damaged),))
            else:
                store.connection.execute('UPDATE snapshots SET '+location+'=?', (encode(damaged),))
    before = path.read_bytes()
    report = db.check_database(path, 'scope')
    assert not report['ok']
    assert next(i['count'] for i in report['errors'] if i['code'] == 'state_payload_invalid') == 1
    assert path.read_bytes() == before


def seed_records(store, labels, identity='input'):
    for model in ('wd', 'dtq'):
        store.put('1', model, {'markers': segment(model, [[(label, .9) for label in labels]]*6, identity)})


def test_merge_cleanup_and_partial_publication_recovery_match_old_snapshots(tmp_path, monkeypatch):
    variants = []
    for compact_storage in (False, True):
        path = tmp_path/('compact.sqlite3' if compact_storage else 'old.sqlite3')
        with StateStore(path, merge_store.scope_for(app.STASH_URL), cache_bytes=2**20) as store:
            if not compact_storage:
                # Both variants use v2; one deliberately skips proof interning.
                monkeypatch.setattr(store, 'pack_snapshot', lambda value: store.pack({FORMAT_KEY: FORMAT, 'snapshot': value, 'proofs': []}))
            stash = Stash()
            resolver = app.services.TagResolver(stash, source=merge_labels.AI)
            policies = empty_policies()
            seed_records(store, ['core', 'abc', 'def'])
            hits, spans = merge.evidence(app, store, '1', policies, policies)
            initial = merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), hits, spans)
            snapshot = store.snapshot('1', 'markers')[0]
            stored = json.loads(zlib.decompress(store.connection.execute("SELECT published FROM snapshots WHERE kind='markers'").fetchone()[0]))
            assert stored[FORMAT_KEY] == FORMAT
            # Publication enriches provenance with build options; marker outputs stay identical.
            markers = lambda items: [{k: v for k, v in item.items() if k != 'contributors'}
                                     for item in merge_intervals.union_intervals(items)]
            assert markers(snapshot['contributions']) == markers(spans)
            repeated = merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), hits, spans)
            assert repeated['markers_unchanged'] == 3
            raw_results = store.connection.execute('SELECT data FROM results ORDER BY model').fetchall()
            cores = {**policies, 'wd': frozenset({'core'})}
            cleaned = cleanup_evidence.cleanup_support(app, store, '1', snapshot['contributions'], {'wd'}, cores, policies)
            cleaned_stats = merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), [], cleaned,
                                             cleanup=True, migrate=False, tag_policies=policies)
            assert store.connection.execute('SELECT data FROM results ORDER BY model').fetchall() == raw_results

            seed_records(store, ['core', 'abc', 'def', 'added', 'new'], identity='next-input')
            hits, spans = merge.evidence(app, store, '1', policies, policies)
            create, attempts = stash.create_marker, 0
            def fail_second(payload):
                nonlocal attempts
                attempts += 1
                if attempts == 2:
                    raise RuntimeError('partial publication')
                return create(payload)
            stash.create_marker = fail_second
            with pytest.raises(RuntimeError, match='partial publication'):
                merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), hits, spans)
            pending = store.connection.execute("SELECT pending FROM snapshots WHERE kind='markers'").fetchone()[0]
            recovery.abandon(store, lambda _: None)
            assert store.connection.execute("SELECT data FROM recovery_publications WHERE kind='markers'").fetchone()[0] == pending
            observed, reconciled = recovery_recovery.reconcile(store, stash.find_scene('1'), 'markers',
                                                      store.snapshot('1', 'markers')[0])
            assert reconciled and 'new' not in {s['tag'] for s in observed['contributions']}
            stash.create_marker = create
            finished = merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), hits, spans)
            assert not store.has_pending_publications() and not recovery.publications(store, '1', 'markers')
            variants.append((stash.scene, stash.writes, initial, repeated, cleaned_stats, observed, finished,
                             store.snapshot('1', 'markers')[0]))
        assert db.check_database(path, merge_store.scope_for(app.STASH_URL))['ok']
    assert variants[0] == variants[1]
