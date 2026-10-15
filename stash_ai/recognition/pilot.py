"""Developer evaluation of bounded, explicitly labeled retained face crops."""
from stash_ai.runtime.cleanup import closing
import json
import sqlite3
import time
from types import SimpleNamespace

from .rebuild import rebuild_selected
from .database import SCHEMA, encoded
from .evaluation import evaluate_pairs
from .portrait_availability import retained_image
from .models import ensure_assets, load_backend, namespace, normalized_embedding, verify, ASSETS


def evaluate_observations(database, run, configurations, labels, *, assets=ensure_assets, backend=load_backend):
    """No Stash/media calls or published membership changes; results stay in SQLite.

    Labels map selected retained observation IDs to explicit human ground truth.
    Compare both adapters on exactly the same crops/landmarks. Only this annotated
    selection is measured; supplied thresholds are model-specific, not calibrated
    automatically. Peak/model loading performance belongs to the isolated smoke
    benchmark rather than being misattributed across models in this process.
    """
    if not isinstance(labels, dict) or not 1 <= len(labels) <= 128:
        raise ValueError('Pilot requires 1..128 explicitly labeled observation IDs.')
    if any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() or len(v) > 100 for k,v in labels.items()):
        raise ValueError('Pilot labels must be explicit bounded nonempty strings.')
    context = database.connection.execute('SELECT scope,selection FROM runs WHERE id=?', (run,)).fetchone()
    if context is None:
        raise ValueError('Pilot results require a face SQLite run history context.')
    selection = json.loads(context['selection'])
    records, retained = [], 0
    for identifier in sorted(labels):
        record = database.connection.execute('SELECT o.*,s.scope,s.scene FROM observations o JOIN scans s ON s.id=o.scan '
            'WHERE o.id=?', (identifier,)).fetchone()
        if record is None or record['scope'] != context['scope'] or record['scene'] not in selection:
            raise ValueError('Pilot observations must belong to this declared selected-scene scope.')
        retained += len(record['crop'] or b'')
        if retained > 33554432:
            raise ValueError('Pilot retained crop byte budget exceeded.')
        records.append(record)
    versions = {}
    for record in records:
        prior = versions.setdefault(record['scene'], record['scan'])
        if prior != record['scan']:
            raise ValueError('Pilot requires one retained scan revision per scene.')
    database.progress(run, 'pilot_ground_truth', {'labels': labels, 'observations': len(records)})
    results = {}
    for name, config in configurations.items():
        if name != config.data['models']['recognizer']:
            raise ValueError('Pilot adapter name/configuration mismatch.')
        paths, vectors, skipped = assets(config), {}, {}
        started = time.perf_counter()
        adapter = backend(config, paths)
        try:
            for row in records:
                if row['crop'] is None or row['rejected'] is not None:
                    skipped[row['id']] = 'no_retained_eligible_crop'
                    continue
                image, face = retained_image(row)
                with image:
                    vectors[row['id']] = normalized_embedding(adapter.embed(image, face), adapter.dimensions)
        finally:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('close pilot identity adapter', adapter.close)])
        embedding_seconds = time.perf_counter() - started
        identifiers = sorted(vectors)
        pairs = ((vectors[a], vectors[b], labels[a] == labels[b]) for i,a in enumerate(identifiers) for b in identifiers[i+1:])
        pair_metrics = evaluate_pairs(pairs, config.data['recognition']['match_threshold'])
        # Run the production selected-scene replay in disposable in-memory SQLite,
        # never changing the operator's retained published identities/membership.
        with closing(sqlite3.connect(':memory:')) as temporary:
            temporary.row_factory = sqlite3.Row
            temporary.executescript(SCHEMA)
            pilot = SimpleNamespace(connection=temporary, installation=database.installation)
            selected = []
            for scene in sorted({r['scene'] for r in records}, key=int):
                scan = 'pilot-' + scene
                rows = [r for r in records if r['scene'] == scene and r['id'] in vectors]
                temporary.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (scan, context['scope'], scene, namespace(config), 'pilot', 'pilot', '{}', '{}', len(rows), len(rows), 'complete'))
                for row in rows:
                    vector = vectors[row['id']]
                    temporary.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        (row['id'], scan, row['cue'], row['seconds'], row['source'], row['rect'], row['geometry'], row['quality'], None,
                         len(vector), vector.astype('<f4').tobytes(), None))
                temporary.commit()
                selected.append({'scene': scene, 'scan': scan})
            rebuild_selected(pilot, selected, config)
            assignments = dict(temporary.execute('SELECT observation,person FROM members'))
            ambiguous = temporary.execute("SELECT COUNT(*) FROM candidates WHERE json_extract(payload,'$.reason')='ambiguous_identity'").fetchone()[0]
        false_merge = false_miss = 0
        for i, first in enumerate(identifiers):
            for second in identifiers[i+1:]:
                same = labels[first] == labels[second]
                grouped = first in assignments and second in assignments and assignments[first] == assignments[second]
                false_merge += int(grouped and not same)
                false_miss += int(same and not grouped)
        artifacts, unavailable = {}, {}
        required_assets = {config.data['models']['detector'], config.data['models']['recognizer']}
        if config.data['detection']['min_visible_face_fraction'] > 0:
            required_assets.add('faceocc')
        for key, path in paths.items():
            try:
                if key in ASSETS:
                    verify(path, ASSETS[key])
                artifacts[key] = path.stat().st_size
            except (OSError, ValueError) as error:
                if key in required_assets:
                    raise
                unavailable[key] = type(error).__name__
        results[name] = {'namespace': namespace(config), 'config': config.snapshot(),
            'scope': 'supplied-selected-retained-crops-only', 'observations': len(records),
            'embedding_coverage': len(vectors), 'accepted_coverage': len(assignments),
            'withheld': len(vectors) - len(assignments), 'ambiguous_observations': ambiguous,
            'pair_false_merge': false_merge, 'pair_false_miss_including_withheld': false_miss,
            'embedding_pair_metrics': pair_metrics, 'skipped': skipped,
            'load_and_embedding_seconds': embedding_seconds,
            'artifact_bytes': sum(artifacts.values()), 'artifact_sizes': artifacts,
            'unavailable_optional_artifacts': unavailable,
            'ram': 'measure separately with isolated benchmark; shared-process peak is not per-adapter'}
    database.progress(run, 'pilot_comparison', results)
    return results
