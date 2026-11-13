"""Source-context navigation restricted to a retained scope and model."""
from contextlib import closing
import json
from .inspection_runtime import open_readonly
from .body_observations import available, contract
from .models import namespace


def cue_geometry(config, scan, cue, selected_kind, selected_id):
    shapes=[]
    with closing(open_readonly(config.database)) as db:
        for row in db.execute('SELECT id,geometry FROM observations WHERE scan=? AND cue=? ORDER BY id',(scan,cue)):
            shapes.append(dict(kind='face',id=row['id'],selected=selected_kind=='face' and selected_id==row['id'],geometry=json.loads(row['geometry']),context=True))
        if available(db):
            for row in db.execute('SELECT id,geometry FROM body_observations WHERE scan=? AND cue=? AND contract=? ORDER BY id',(scan,cue,contract(config))):
                shapes.append(dict(kind='body',id=row['id'],selected=selected_kind=='body' and selected_id==row['id'],geometry=json.loads(row['geometry']),context=True))
            if selected_kind=='body' and not any(s['selected'] for s in shapes):
                row=db.execute('SELECT geometry FROM body_observations WHERE id=? AND scan=? AND cue=?',(selected_id,scan,cue)).fetchone()
                if row:shapes.append(dict(kind='body',id=selected_id,selected=True,geometry=json.loads(row['geometry']),context=True))
    return shapes


def locate(config, kind, address, scope):
    from .identity_inspection import IdentityReader, TABLES
    from .scene_inspection import UnassignedReader
    if kind not in TABLES:raise ValueError('Kind must be face or body.')
    observations,members=TABLES[kind]
    with closing(open_readonly(config.database)) as db:
        row=db.execute(f'SELECT o.id,o.cue,o.seconds,s.id AS scan,s.scene,s.scope,s.namespace,s.policy_sha,s.status '
            f'FROM {observations} o JOIN scans s ON s.id=o.scan WHERE o.id=? AND s.scope=? AND s.namespace=?',
            (address,scope,namespace(config))).fetchone()
        if row is None:raise ValueError('Observation unavailable in this scope/model.')
        scan=dict(row);scan['id']=scan.pop('scan')
        owner=db.execute(f'SELECT person FROM {members} WHERE observation=?'+(" AND status='accepted'" if kind=='body' else ''),(address,)).fetchone()
    reader=IdentityReader(config,identity_id=owner[0],scope=scope) if owner and owner[0] else UnassignedReader(config,scan)
    rows,_=reader.page(kind=kind,observation=address)
    if not rows:raise ValueError('Observation changed while navigating; refresh.')
    return dict(rows[0],reader=reader)
