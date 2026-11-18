"""Read-only, bounded payload sampling for an explicitly sized VTT library."""
import json
import math
from pathlib import Path


def estimate(db,config,*,scenes=10000,cues=300,positive_fraction=.7,faces=2.,bodies=2.,sample_scenes=32,overhead=1.35):
    from .scan_heads import predicate
    from .cold_payloads import available,length_sql
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    from .models import namespace
    selected_namespace=namespace(config)
    current=list(db.execute("SELECT s.id,s.scene FROM scans s WHERE s.status='complete' AND s.namespace=? AND "+predicate()+" ORDER BY s.scope,s.namespace,CAST(s.scene AS INTEGER)",(selected_namespace,)))
    # Evenly spaced scenes, with bounded rows per scene; no image/model loading.
    selected=[current[i] for i in sorted({int(i*len(current)/min(sample_scenes,len(current))) for i in range(min(sample_scenes,len(current)))})] if current else []
    target_frames=scenes*cues
    report=dict(model_namespace=selected_namespace,assumptions=dict(scenes=scenes,cues_per_scene=cues,positive_fraction=positive_fraction,
                faces_per_positive_cue=faces,bodies_per_positive_cue=bodies,sqlite_overhead_factor=overhead),
                target_frames=target_frames,sampled_scenes=[r[1] for r in selected],
                current_complete_scans=len(current),components={})
    totals=[]
    for table,kind,per_frame in [('observations','faces',faces),('body_observations','bodies',bodies)]:
        if table not in tables:continue
        crop=length_sql('o.crop') if available(db) else 'length(o.crop)'
        mask="COALESCE((SELECT length(mask) FROM body_quality_masks WHERE observation=o.id),0)" if kind=='bodies' and 'body_quality_masks' in tables else '0'
        fields=['COALESCE(length(o.embedding),0)',f'COALESCE({crop},0)',mask,
                'length(CAST(o.geometry AS BLOB))+length(CAST(o.quality AS BLOB))']
        rows=[]
        for scan,_ in selected:
            rows.extend(db.execute(f'SELECT {",".join(fields)} FROM {table} o WHERE scan=? ORDER BY cue,id LIMIT 1000',(scan,)))
        count=math.ceil(target_frames*positive_fraction*per_frame)
        parts={name:sum(r[i] for r in rows)/len(rows) for i,name in enumerate(('embedding','crop','mask','metadata'))} if rows else None
        mean=sum(parts.values()) if parts else None
        projected=math.ceil(count*mean*overhead) if mean is not None else None
        report['components'][kind]=dict(sample_rows=len(rows),target_rows=count,mean_payload_bytes=parts,
            projected_bytes=projected,projected_at_largest_sample_bytes=math.ceil(count*max(map(sum,rows))*overhead) if rows else None)
        if projected is not None:totals.append(projected)
    complete=db.execute("SELECT COUNT(*) FROM scans WHERE status='complete' AND namespace=?",(selected_namespace,)).fetchone()[0]
    history=max(1.,complete/max(1,len(current)))
    report['observed_scan_history_factor']=history
    report['projected_current_evidence_bytes']=sum(totals) if len(totals)==2 else None
    report['projected_with_observed_history_bytes']=math.ceil(sum(totals)*history) if len(totals)==2 else None
    ex=config.data['execution']
    report['configured_reserves_bytes']={k:ex.get(k,0) for k in ('rebuild_scratch_bytes','rebuild_checkpoint_bytes','inference_cache_bytes','derived_cache_bytes','media_cache_bytes','disk_free_bytes')}
    report['limits']=['Payload estimates include rejected rows and resolve cold crop addresses; duplicate crops may overestimate storage.',
        'Sampled scenes may not represent the library. Largest-sample values are sensitivity scenarios, not confidence intervals.',
        'SQLite overhead is an explicit assumption. Identity indices, model assets, publications, WAL peak and extra history require separate headroom.',
        'Configured cache/scratch reserves are ceilings, not measured simultaneous usage. No runtime or GPU throughput is inferred.']
    return report


def execute(args,config):
    from .db_maintenance import _path,_connect,_header,_files
    path=_path(config.database)
    with _connect(path) as db:
        db.execute('BEGIN');_header(db)
        report=estimate(db,config,scenes=args.target_scenes,cues=args.cues_per_scene,
            positive_fraction=args.positive_fraction,faces=args.faces_per_cue,bodies=args.bodies_per_cue,
            sample_scenes=args.sample_scenes,overhead=args.sqlite_overhead)
        report['current_files_bytes']=_files(path)
    print(json.dumps(report,indent=2))
    return 0
