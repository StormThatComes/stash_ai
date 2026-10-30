"""Fingerprint the scoped retained inputs consumed by replay, without images."""
import hashlib
import json
from .database import encoded


def fingerprint(db, scope, namespace):
    # Match copy_history: every version of a scoped scan can affect origin
    # precedence, current-head admission or an external historical rival.
    scan='SELECT id FROM scans WHERE scope=? AND namespace=?'
    people='SELECT id FROM people WHERE scope=? AND namespace=?'
    observations=f'SELECT id FROM observations WHERE scan IN ({scan})'
    bodies=f'SELECT id FROM body_observations WHERE scan IN ({scan})'
    scoped=(scope,namespace)
    rules={
        'scans': ('id IN ('+scan+')', scoped),
        'observations': ('scan IN ('+scan+')', scoped),
        'person_evidence': ('observation IN ('+observations+')', scoped),
        'body_observations': ('scan IN ('+scan+')', scoped),
        'body_cues': ('scan IN ('+scan+')', scoped),
        'people': ('id IN ('+people+')', scoped),
        'members': (f'observation IN ({observations}) AND person IN ({people})', scoped*2),
        'candidates': ('scan IN ('+scan+')', scoped),
        'identity_origins': (f'observation IN ({observations}) AND person IN ({people})', scoped*2),
        'identity_merges': (f'source IN ({people}) AND target IN ({people})', scoped*2),
        'merge_support': (f'observation IN ({observations}) AND source IN ({people}) AND target IN ({people})', scoped*3),
        'body_members': (f'observation IN ({bodies}) AND (person IS NULL OR person IN ({people}))', scoped*2),
        'body_origins': (f'observation IN ({bodies}) AND person IN ({people})', scoped*2),
        'evidence_thumbnails': ("id IN (SELECT json_extract(CASE WHEN json_valid(quality) THEN quality ELSE '{}' END,"
            f"'$.evidence_thumbnail_id') FROM observations WHERE scan IN ({scan}))", scoped),
        'metadata': ("key='installation' OR (key LIKE 'recognition-scan-head:%' AND "
            f"json_extract(value,'$.scan') IN ({scan}) AND json_extract(value,'$.latest_created_scan') IN ({scan}))", scoped*2),
    }
    result=hashlib.sha256()
    def add(value):
        raw=b'b'+value if isinstance(value,bytes) else b'j'+json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        result.update(str(len(raw)).encode()+b':'+raw)
    for table in sorted(rules):
        schema=db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone()
        add(table)
        if schema is None:
            add(None);continue
        add(schema[0])
        columns=[r[1] for r in db.execute(f'PRAGMA table_info({table})') if r[1]!='crop']
        where,args=rules[table]
        for row in db.execute(f'SELECT {",".join(columns)} FROM {table} WHERE {where} ORDER BY rowid',args):
            add('row')
            for value in row:add(value)
    return result.hexdigest()
