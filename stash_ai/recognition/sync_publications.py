"""Explicit finite publication of queued current memberships; no inference."""
from collections import Counter
from contextlib import closing as close_database
import json
import os
from pathlib import Path

from .database import FaceDatabase
from .fragmentation_report import _open_readonly
from .models import namespace
from .publication_queue import plan,inventory,validate,adopt
from .run import instance_scope,_publish_selection
from .stash import RecognitionStash
from .progress import WorkProgress
from stash_ai.runtime.cleanup import closing
from stash_ai.runtime.console import present
from stash_ai.runtime.report import RunReport


def execute(args,config,*,services=None):
    path=Path(config.database)
    if not path.is_file():raise FileNotFoundError('Recognition synchronization requires an existing face database.')
    url=os.getenv('STASH_URL','http://localhost:9999');api_key=os.getenv('STASH_API_KEY','')
    scope=instance_scope(url);model_namespace=namespace(config)
    with close_database(_open_readonly(path)) as db:
        entries=plan(db,scope,model_namespace,config,getattr(args,'scene_id',()) or ())
        document=dict(format=1,kind='recognition_publication_sync',scope=scope,namespace=model_namespace,
                      counts=inventory(db,scope,model_namespace),scenes=entries)
    if not args.apply:
        print(json.dumps(document,indent=2,sort_keys=True));return 0
    from tqdm import tqdm
    report=RunReport(args.progress_style=='verbose',tqdm.write);counts=Counter()
    client_factory=getattr(services,'client',RecognitionStash)
    with FaceDatabase(path) as database:
        run=database.start_run(scope,[e['scene'] for e in entries],
            {**config.snapshot(),'invocation':{'command':'sync-identities','apply':True}})
        outcome='success';failure=None
        try:
            client=client_factory(url=url,api_key=api_key,timeout=config.data['execution']['timeout_seconds'])
            with closing(client,report=report,name='close recognition synchronization HTTP'),present(report,args.progress_style):
                client.report=report
                with WorkProgress(report,'Synchronize recognition links',len(entries)) as progress:
                    for entry in entries:
                        if entry['status']!='ready':
                            counts['scenes_blocked']+=1
                            report.issue('recognition','synchronization',entry['scene'],ValueError(entry['reason']),entry['reason'])
                            progress.advance(blocked=counts['scenes_blocked']);continue
                        try:
                            from .scene_sync import reconcile_unlinks,reconcile_legacy_merges
                            with database.connection:adopt(database.connection,scope,model_namespace,config,entry)
                            validate(database.connection,scope,model_namespace,entry)
                            reconcile_legacy_merges(database,client,scope,[entry['scene']])
                            reconcile_unlinks(database,client,scope,entry['scene'])
                        except Exception as error:
                            from .diagnostics import message
                            counts['people_failed']+=1
                            report.issue('recognition','synchronization',entry['scene'],error,message(error,api_key))
                            progress.advance(failed=counts['people_failed']);continue
                        item=dict(scene=entry['scene'],scan=entry['scan'])
                        groups={item['scan']:[dict(person=p,status='accepted',observations=0) for p in entry['people']]}
                        def guard(_item):validate(database.connection,scope,model_namespace,entry)
                        _publish_selection(database,client,report,config,scope,model_namespace,[item],groups,{},counts,
                                           api_key,run,progress,guard=guard)
                if counts['scenes_blocked'] or counts['people_failed']:outcome='partial'
                database.progress(run,'synchronization',dict(counts))
                report.console.finish_rendering()
                report.write('Recognition synchronization '+outcome+': '+
                    ' '.join(f'{k}={v}' for k,v in sorted(counts.items())))
                return 0 if outcome=='success' else 1
        except BaseException as error:
            from .diagnostics import message
            outcome='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            failure=message(error,api_key)[:1000];raise
        finally:
            database.finish_run(run,outcome,{'counts':dict(counts)},failure)
