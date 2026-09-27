"""Synthetic SQLite benchmark. Creates a temporary DB; never connects to Stash.

python -m benchmarks.benchmark_cleanup_sql --markers 2500000 > cleanup_benchmark.json
Measures discovery only; Stash API, network, hooks and file deletion are excluded.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stash_ai.cleanup_sql import SQL, SQLCleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markers', type=int, default=2_500_000)
    parser.add_argument('--legacy-time-budget', type=float, default=5)
    args = parser.parse_args()
    if args.markers < 1 or args.legacy_time_budget <= 0:
        parser.error('Both limits must be positive.')
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix='stash-cleanup-benchmark-') as directory:
        db = sqlite3.connect(Path(directory)/'synthetic.sqlite3')
        try:
            # These settings apply ONLY to the freshly-created, disposable fixture.
            db.executescript('''PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                CREATE TABLE tags(id INTEGER PRIMARY KEY, name TEXT UNIQUE);
                CREATE TABLE scene_markers(id INTEGER PRIMARY KEY, title TEXT, primary_tag_id INTEGER, scene_id INTEGER);
                CREATE TABLE scene_markers_tags(scene_marker_id INTEGER, tag_id INTEGER);
                CREATE TABLE scenes_tags(scene_id INTEGER, tag_id INTEGER);
                INSERT INTO tags VALUES (1,'ai:blocked'),(2,'ai:keep'),(3,'ai:marker:stash-taggers');''')
            db.execute('''WITH RECURSIVE numbers(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM numbers WHERE n < ?)
                INSERT INTO scene_markers SELECT n, '[AI] fixture', CASE WHEN n % 100=0 THEN 1 ELSE 2 END,
                    1 + (n-1)/300 FROM numbers''', (args.markers,))
            db.executescript('''INSERT INTO scene_markers_tags SELECT id, 1 FROM scene_markers;
                INSERT INTO scene_markers_tags SELECT id, 3 FROM scene_markers;
                CREATE INDEX index_scene_markers_on_primary_tag_id ON scene_markers(primary_tag_id);
                CREATE INDEX index_scene_markers_on_scene_id ON scene_markers(scene_id);
                CREATE INDEX index_scene_markers_tags_on_tag_id ON scene_markers_tags(tag_id);
                CREATE INDEX index_scene_markers_tags_on_scene_marker_id ON scene_markers_tags(scene_marker_id);
                CREATE INDEX index_scenes_tags_on_tag_id ON scenes_tags(tag_id);''')
            db.commit()
            result = {'sqlite_version': sqlite3.sqlite_version, 'markers': args.markers,
                      'extra_associations': args.markers*2, 'page_size': 2000,
                      'fixture_seconds': round(perf_counter()-started, 3),
                      'scope': 'Synthetic local SQLite discovery only; no Stash API mutations or network.'}
            class Reader:
                def gql(self, query, variables):
                    return {'querySQL': {'rows': [list(row) for row in db.execute(variables['sql'],variables['args'])]}}
            stats = Counter()
            engine = SQLCleanup(SimpleNamespace(stash=Reader()),
                SimpleNamespace(cleanup_page_size=2000,cleanup_write_batch_size=200), stats, None)
            engine.probe()
            for name, prefix, owner in [('global_primary','', ''), ('owned_core','[AI] ', 'ai:marker:stash-taggers')]:
                stats.clear()
                started = perf_counter()
                params = (1,'ai:blocked',prefix,prefix,prefix,owner)
                count = sum(len(page) for page in engine.pages('primary',params,engine.bounds[0]))
                result[name] = {'rows': count, 'queries': stats['cleanup_sql_reads'],
                                'seconds': round(perf_counter()-started, 4)}
            plans = {}
            for name, key, values in [('primary','primary',(1,'ai:blocked','','','','',0,args.markers,2000)),
                ('owned_core','primary',(1,'ai:blocked','[AI] ','[AI] ','[AI] ','ai:marker:stash-taggers',0,args.markers,2000)),
                ('extras_tail','extras',(1,'ai:blocked',max(0,args.markers-2000),args.markers*2,2000))]:
                plans[name] = [row[3] for row in db.execute('EXPLAIN QUERY PLAN '+SQL[key],values)]
            result['query_plans'] = plans
            # Same costly match shape as the old marker tag filter: union primary
            # and extra references, then count all matches again for each page.
            old = '''WITH marker_tags AS (
                SELECT scene_marker_id FROM scene_markers_tags WHERE tag_id=1
                UNION SELECT id FROM scene_markers WHERE primary_tag_id=1)
                SELECT count(*) FROM scene_markers m LEFT JOIN marker_tags mt
                ON mt.scene_marker_id=m.id WHERE mt.scene_marker_id IS NOT NULL'''
            started = perf_counter()
            db.set_progress_handler(lambda: int(perf_counter()-started > args.legacy_time_budget), 10000)
            try:
                total = db.execute(old).fetchone()[0]
                result['legacy_one_count'] = {'rows':total,'seconds':round(perf_counter()-started,4),'completed':True}
            except sqlite3.OperationalError as exc:
                if 'interrupted' not in str(exc):
                    raise
                result['legacy_one_count'] = {'seconds':round(perf_counter()-started,4),'completed':False,
                                              'stopped_after_seconds':args.legacy_time_budget}
            finally:
                db.set_progress_handler(None,0)
            result['legacy_matching_rows'] = args.markers
            result['legacy_pages_at_1000'] = (args.markers+999)//1000
            print(json.dumps(result,indent=2))
        finally:
            db.close()


if __name__ == '__main__':
    main()
