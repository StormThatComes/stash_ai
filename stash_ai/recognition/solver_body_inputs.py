"""Explain missing body evidence from selected goal cues without reading vectors."""
from .body_observations import available, contract
from .database import encoded


def diagnose(db, config, ids, selected):
    if not ids or not available(db):
        return {}
    args = (encoded(sorted(ids)), encoded(sorted(selected)), contract(config))
    rows = db.execute("""
        WITH wanted AS (SELECT value AS id FROM json_each(?)),
        selected AS (SELECT value AS scan FROM json_each(?)),
        settings AS (SELECT ? AS contract),
        addresses AS (
            SELECT o.id,o.scan,o.cue FROM wanted w JOIN observations o USING(id)
            UNION SELECT b.id,b.scan,b.cue FROM wanted w JOIN body_observations b USING(id)
        )
        SELECT a.id,a.cue,c.cue AS collected,b.id AS detection,b.face,b.rejected,
               b.embedding IS NOT NULL AS ready,json_extract(b.quality,'$.quality.query_only') AS auxiliary
        FROM addresses a JOIN selected USING(scan) CROSS JOIN settings s
        LEFT JOIN body_cues c ON c.scan=a.scan AND c.cue=a.cue AND c.contract=s.contract
        LEFT JOIN body_observations b ON b.scan=a.scan AND b.cue=a.cue AND b.contract=s.contract
        ORDER BY a.id
        """, args)
    result = {}
    nearby_rejections = {}
    for row in rows:
        oid = row['id']
        linked = oid in (row['detection'],row['face'])
        if linked:
            reason = (row['rejected'] or ('embedding_missing' if not row['ready'] else
                      'query_only' if row['auxiliary'] else 'ready_for_matching'))
            result[oid] = (reason, row['cue'])
        else:
            nearby_rejections.setdefault(oid, set()).add(row['rejected'])
        if oid not in result:
            reason = ('body_cue_not_collected' if row['collected'] is None else
                      'no_body_detection' if row['detection'] is None else 'face_body_unassociated')
            result[oid] = (reason, row['cue'])
    for oid, reasons in nearby_rejections.items():
        if result[oid][0] == 'face_body_unassociated' and len(reasons) == 1 and None not in reasons:
            result[oid] = ('unassociated_cue_' + next(iter(reasons)), result[oid][1])
    return result
