"""Shared offline helpers; no test collection or fixture registration."""
from collections import Counter
from copy import deepcopy

from stash_ai.cleanup.prune import REQUIRED_COUNTS


class PruneStash:
    def __init__(self,dry=False):
        self.dry_run=dry;self.tags={};self.deleted=[];self.calls=Counter()
        self.counts=set(REQUIRED_COUNTS)
        self.before_refresh=None
    def add(self,name,parents=(),**counts):
        identifier=str(len(self.tags)+1)
        self.tags[identifier]=dict(id=identifier,name=name,parents=[{"id":p} for p in parents],**{key:0 for key in self.counts})
        self.tags[identifier].update(counts)
        return identifier
    def tag(self,identifier):
        result=deepcopy(self.tags.get(identifier))
        if result is not None:
            result["children"]=[{"id":i} for i,t in self.tags.items() if identifier in {p["id"] for p in t["parents"]}]
            result["child_count"]=len(result["children"])
        return result
    def gql(self,query,variables=None):
        variables=variables or {}
        if "PruneCapabilities" in query:
            self.calls["capabilities"]+=1
            fields=self.counts|{"id","name","description","children","parents","parent_count","movie_count"}
            return {"tag":{"fields":[{"name":n} for n in fields]},
                    "mutation":{"fields":[{"name":"tagDestroy"}]}}
        if "PruneCatalog" in query:
            self.calls["catalog"]+=1
            page=variables["filter"]["page"];size=variables["filter"]["per_page"]
            ids=list(self.tags)[(page-1)*size:page*size]
            return {"findTags":{"tags":[self.tag(i) for i in ids]}}
        if query.lstrip().startswith("mutation"):
            assert not self.dry_run
            identifier=variables["input"]["id"]
            current=self.tag(identifier)
            assert current is not None and not current["children"]
            assert all(current[key]==0 for key in self.counts)
            del self.tags[identifier];self.deleted.append(identifier)
            return {"tagDestroy":True}
        self.calls["refresh"]+=1
        identifier=variables["id"]
        if self.before_refresh:self.before_refresh(self,identifier)
        return {"findTag":self.tag(identifier)}
    def close(self):pass
