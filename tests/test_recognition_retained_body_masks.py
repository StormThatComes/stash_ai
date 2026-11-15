"""Retained SAM pixels follow the observation through collection, replay and migration."""
from contextlib import closing
import sqlite3
from types import SimpleNamespace
import zlib

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.body_collection import collect
from stash_ai.recognition.body_observations import save_records
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.inspection_overlays import render
from tests.test_recognition_body_mask_overlap import person
from tests.test_recognition_inspection_overlays import linked_fixture


@pytest.mark.parametrize('retain', [True, False])
def test_collection_stores_exact_quality_pixels_for_accepted_and_rejected_bodies(tmp_path, retain):
    from tests.test_recognition_body_collection import settings
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.inspection_crops import VttPreviews
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.run import instance_scope
    from stash_ai.recognition.scanning import scan_scene
    from tests.helpers.recognition import Backend, MediaClient

    config = settings(tmp_path)
    data = config.data['sapiensid']
    data['collection'].update(min_keypoints=2, min_person_pixels=1)
    data['storage'].update(retain_person_crops=False, retain_body_masks=retain)
    scan_media = MediaClient(count=1)
    client = SimpleNamespace(url=scan_media.url, media=scan_media.media, scene=lambda _: scan_media.scene, close=lambda: None)
    scope = instance_scope(client.url)
    coords = np.indices((15, 13))
    pixels = (coords[0] >= 3) & (coords[1] < 7)
    ready = person([10, 20, 23, 35], pixels)
    rejected = person([40, 10, 53, 25], ~pixels)
    rejected['keypoints'][1:] = [[-1, -1]] * 16
    rejected['pose_refinement']['encoder_mask'] = rejected['pose_refinement']['overlap_mask']
    missing = person([70, 40, 83, 55])
    missing['pose_refinement'] = dict(status='skipped', reason='no_reliable_prompts')
    model = SimpleNamespace(frame_people=[ready, rejected, missing], face_matches=[], person_geometry=lambda *args: None)
    batch = []
    pipeline = SimpleNamespace(collect_items=lambda _, image, items: batch.extend(r for r, _ in items))
    with FaceDatabase(config.database) as state, SpriteInput(scan_media, scan_media.scene, config) as source:
        db = state.connection
        scan, _ = scan_scene(state, scope, scan_media.scene, config, namespace(config), source, Backend)
        with source.thumbnail(0) as frame:
            config.data['execution']['scan_cpu_workers'] = 1
            collect(pipeline, db, model, config, scan, 0, 0., frame, [], [])
        with db:
            save_records(db, config, batch)
        records = db.execute('SELECT b.id,b.geometry,b.quality,b.rejected,m.mask FROM body_observations b '
                             'LEFT JOIN body_quality_masks m ON m.observation=b.id ORDER BY b.rowid').fetchall()
        assert [r['rejected'] for r in records] == [None, 'person_keypoints', None]
        for row, expected in zip(records, (pixels, ~pixels, None)):
            assert 'overlap_mask' not in row['geometry'] + row['quality']
            assert 'encoder_mask' not in row['geometry'] + row['quality']
            if retain and expected is not None:
                assert zlib.decompress(row['mask']) == np.packbits(expected).tobytes()
            else:
                assert row['mask'] is None
        assert db.execute('SELECT COUNT(*) FROM body_cues WHERE contract=?', (contract(config),)).fetchone()[0] == 1
        # A failed replacement restores the mask and body row together.
        if retain:
            rewritten = [(oid, {k: v for k, v in meta.items() if k != '_quality_mask'}, vector, crop)
                         for oid, meta, vector, crop in batch]
            save_records(db, config, rewritten)
            assert db.execute('SELECT COUNT(*) FROM body_quality_masks').fetchone()[0] == 0
            db.rollback()
            assert db.execute('SELECT COUNT(*) FROM body_quality_masks').fetchone()[0] == 2

    previews = VttPreviews(config, client_factory=lambda: client)
    try:
        geometry = {}
        image, caption = previews.image('body', records[1]['id'], scope, (None, ''), geometry=geometry, full=True)
        assert image is not None and 'pixels verified' in caption
        with image:
            annotated, caption = render(image, geometry, dict(masks=True), database=config.database)
            assert 'No quality mask (no_reliable_prompts)' in caption
            assert ('Recorded quality mask' if retain else 'Mask not retained') in caption
            assert (annotated.getpixel((52, 13)) != image.getpixel((52, 13))) is retain
            assert annotated.getpixel((40, 15)) == image.getpixel((40, 15))
        if retain:
            with FaceDatabase(config.database) as state, state.connection as db:
                db.execute('UPDATE body_observations SET quality=json_set(quality, '
                           "'$.quality.source_pixel_digest', 'wrong') WHERE id=?", (records[1]['id'],))
            image, caption = previews.image('body', records[1]['id'], scope, (None, ''), full=True)
            assert image is None and 'recorded signature' in caption
    finally:
        previews.close()


def test_v17_readonly_preview_and_migration_preserve_old_observations(tmp_path):
    config, reader, _ = linked_fixture(tmp_path)
    with closing(sqlite3.connect(config.database)) as db, db:
        db.execute('DROP TABLE body_quality_masks')
        db.execute('PRAGMA user_version=17')
    geometry = reader.preview('body', 'accepted-body')['geometry']
    with Image.new('RGB', (100, 200)) as image:
        before = config.database.read_bytes()
        _, caption = render(image, geometry, dict(masks=True), database=config.database)
        assert 'Mask not retained' in caption and config.database.read_bytes() == before
        with FaceDatabase(config.database) as state:
            assert state.connection.execute('SELECT COUNT(*) FROM body_quality_masks').fetchone()[0] == 0
        _, caption = render(image, geometry, dict(masks=True), database=config.database)
        assert 'Mask not retained' in caption


def test_mask_decoder_rejects_truncated_and_oversized_payloads():
    from stash_ai.recognition.retained_body_masks import mask_image
    for payload in (zlib.compress(bytes(2))[:-1], zlib.compress(bytes(3)), zlib.compress(bytes(2)) + b'x'):
        with pytest.raises(ValueError):
            mask_image(payload, 5, 3)
