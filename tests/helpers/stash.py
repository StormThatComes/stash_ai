"""Shared offline helpers; no test collection or fixture registration."""

from copy import deepcopy

from stash_ai.publication import separate as separate_sync
from tests.helpers.runtime import runtime as app


class Stash:
    merge_reader = None
    merge_only_reads = False

    def __init__(self, dry=False):
        self.dry_run = dry
        self.create_markers = True
        self.scene_page_size = 100
        self.merge_read_configurations = []
        self.tags, self.writes = {}, []
        self.fail_create = False
        self.scene = dict(id="1", title="fixture", tags=[], scene_markers=[], paths=dict(sprite="/sprite",vtt="/vtt"))
    def tag(self,name):
        if name not in self.tags:
            self.tags[name] = dict(id=str(max((int(t['id']) for t in self.tags.values()), default=0)+1),name=name,parents=[],description=None)
        return self.tags[name]
    def find_tag(self,name): return deepcopy(self.tags.get(name))
    def find_tag_by_id(self,identifier,required=True):
        tag=next((t for t in self.tags.values() if t["id"]==identifier),None)
        if tag is None and required: raise RuntimeError('Missing tag')
        return deepcopy(tag)
    def create_tag(self,name,**kwargs):
        assert not self.dry_run
        self.writes.append(("tag",name))
        tag=self.tag(name)
        tag["parents"]=[self.find_tag_by_id(i) for i in kwargs.get("parent_ids",[])]
        tag["description"]=kwargs.get("description")
        tag["ignore_auto_tag"]=kwargs.get("ignore_auto_tag",False)
        return deepcopy(tag)
    def update_tag_organization(self,identifier,parents,ignore_auto_tag):
        self.update_tag_parents(identifier,parents)
        tag=next(t for t in self.tags.values() if t['id']==identifier)
        tag['ignore_auto_tag']=ignore_auto_tag
        return deepcopy(tag)
    def update_tag_parents(self,identifier,parents):
        assert not self.dry_run
        tag=next(t for t in self.tags.values() if t["id"]==identifier)
        tag["parents"]=[self.find_tag_by_id(i) for i in parents]
        self.writes.append(("parents",identifier))
        return deepcopy(tag)
    def marker(self,source,label,identifier,manual=False):
        marker=dict(id=identifier,title="Manual" if manual else source.title_prefix+label,seconds=0,end_seconds=2,
                    primary_tag=self.tag(source.prefix+label),tags=[] if manual else [self.tag(source.owner_tag)])
        self.scene["scene_markers"].append(marker)
        return marker
    def find_scene(self,identifier): return deepcopy(self.scene)
    def configure_merge_reads(self, report, *, merge_only=False):
        self.report = report
        self.merge_read_configurations.append(merge_only)
    def find_scene_for_merge(self, identifier):
        return self.find_scene(identifier)
    def get_media(self,url): return url.encode("utf-8")
    def iter_scenes(self,identifiers=None): yield self.find_scene("1")
    def iter_tags(self): yield from deepcopy(list(self.tags.values()))
    def update_scene_tags(self,identifier,ids):
        assert not self.dry_run
        self.writes.append(("scene",ids))
        self.scene["tags"]=[self.find_tag_by_id(i) for i in ids]
    def record(self,payload,identifier):
        return dict(id=identifier,title=payload["title"],seconds=payload["seconds"],end_seconds=payload.get("end_seconds"),
                    primary_tag=self.find_tag_by_id(payload["primary_tag_id"]),tags=[self.find_tag_by_id(i) for i in payload["tag_ids"]])
    def create_marker(self,payload):
        assert not self.dry_run
        if self.fail_create: raise RuntimeError("injected failure")
        self.writes.append(("create",payload))
        self.scene["scene_markers"].append(self.record(payload,str(1000+len(self.writes))))
    def update_marker(self,payload):
        assert not self.dry_run
        self.writes.append(("update",payload))
        self.scene["scene_markers"]=[self.record(payload,m["id"]) if m["id"]==payload["id"] else m for m in self.scene["scene_markers"]]
    def delete_marker(self,identifier):
        assert not self.dry_run
        self.writes.append(("delete",identifier))
        self.scene["scene_markers"]=[m for m in self.scene["scene_markers"] if m["id"]!=identifier]
    def close(self): pass
    def gql(self,query,variables=None):
        from stash_ai.cleanup.prune import REQUIRED_COUNTS
        counts=REQUIRED_COUNTS|{'studio_count','group_count'}
        if 'querySQL' in query:
            raise RuntimeError('SQL temporarily unavailable')
        def usage(tag):
            identifier=tag['id']
            scenes=getattr(self,'scenes',{'1':self.scene}).values()
            children=[{'id':t['id']} for t in self.tags.values() if any(p['id']==identifier for p in t['parents'])]
            return {**deepcopy(tag),**{name:0 for name in counts},'children':children,'child_count':len(children),
                    'scene_count':sum(any(t['id']==identifier for t in s['tags']) for s in scenes),
                    'scene_marker_count':sum(any(t['id']==identifier for t in [m['primary_tag'],*m['tags']])
                                             for s in scenes for m in s['scene_markers']),
                    'image_count':sum(any(t['id']==identifier for t in image['tags']) for image in getattr(self,'images',[]))}
        if 'PruneCapabilities' in query:
            return {'tag':{'fields':[{'name':n} for n in counts|{'id','name','description','parents','children'}]},
                    'studio':{'fields':[{'name':'tags'}]},'group':{'fields':[{'name':'tags'}]},
                    'movie':None,'mutation':{'fields':[{'name':'tagDestroy'}]}}
        if 'PruneCatalog' in query:
            f=variables['filter']; values=list(self.tags.values()); start=(f['page']-1)*f['per_page']
            return {'findTags':{'count':len(values),'tags':[usage(t) for t in values[start:start+f['per_page']]]}}
        if 'PruneCandidates' in query:
            return {'findTags':{'tags':[usage(t) for t in self.tags.values() if t['id'] in variables['ids']]}}
        if query.lstrip().startswith('mutation'):
            assert not self.dry_run
            tag=self.find_tag_by_id(variables['input']['id'])
            assert all(usage(tag)[name]==0 for name in counts)
            self.writes.append(('delete_tag',tag['name']))
            del self.tags[tag['name']]
            return {'tagDestroy':True}
        tag=self.find_tag_by_id(variables['id'],required=False)
        return {'findTag':usage(tag) if tag is not None else None}


def sync(stash,source,predictions=(),specs=(),cores=frozenset(),scenes=frozenset()):
    return separate_sync.sync_scene(app, stash,app.services.TagResolver(stash,source=source),stash.find_scene("1"),list(predictions),list(specs),cores,source,scenes)

