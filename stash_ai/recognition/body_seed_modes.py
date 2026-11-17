"""Frozen complete-link body appearances of an already known face identity."""


class FrozenSeeds:
    def __init__(self,seeds,scores,settings):
        self.seeds,self.scores,self.settings=seeds,scores,settings
        self.modes={}
        self.support={}
        threshold=max(settings['match_threshold'],settings['cohesion_threshold'])
        for person,rows in seeds.items():
            modes=[]
            for row in rows:
                if len(modes)==1 and scores.certifies(row,modes[0],threshold):
                    modes[0].append(row);continue
                compatible = []
                for index, mode in enumerate(modes):
                    mean, minimum = scores.compare([row],mode)
                    if minimum >= threshold:
                        compatible.append((minimum,mean,len(mode),-index))
                if compatible:
                    _,_,_,negative_index = max(compatible)
                    modes[-negative_index].append(row)
                else:
                    modes.append([row])
            self.modes[person]=modes
            from .body_identity import independent
            for mode in modes:
                self.support[id(mode)] = len(independent(mode,settings))

    def add_scene_anchors(self, additions):
        banks = {owner: [*self.seeds[owner], *extra] for owner, extra in additions.items()}
        expanded = FrozenSeeds(banks, self.scores, self.settings)
        self.seeds = {**self.seeds, **banks}
        for owner, extra in additions.items():
            local_ids = {row['id'] for row in extra}
            for mode in expanded.modes[owner]:
                if any(row['id'] in local_ids for row in mode):
                    # Also retain the original strict mode as a scoring option.
                    self.modes[owner].append(mode)
                    self.support[id(mode)] = expanded.support[id(mode)]

    def rank(self,row,guard):
        from .body_identity import copresent
        settings=self.settings;ranked=[]
        for person,modes in self.modes.items():
            allowed=not copresent([row],self.seeds[person]) and guard.allows(row,self.seeds[person],person)
            if not allowed:continue  # Certified contradiction is not an unknown rival.
            choices=[]
            for mode in modes:
                mean,minimum=self.scores.compare([row],mode)
                good=(allowed and mean>=settings['match_threshold'] and minimum>=settings['cohesion_threshold']
                      and self.support[id(mode)]>=settings['attach_min_support'])
                choices.append((mean,minimum,person,mode,good))
            valid=[c for c in choices if c[4]]
            if not valid and len(choices)>1:
                from .body_identity import independent
                # Every contributing look must beat the eventual rival itself.
                refs=[r for mean,low,_,mode,_ in choices if low>=settings['match_threshold']
                      for r in mode if r.get('face_anchor') and r.get('face_person')==person]
                if len(refs)>=settings['attach_min_support']:
                    refs=independent(refs,settings)
                    if len(refs)>=settings['attach_min_support']:
                        low=min(self.scores.face_similarity(row['_vector'],r['_vector']) for r in refs)
                        valid.append((low,low,person,refs,True))
            chosen=max(valid or choices,key=lambda c:(c[0],c[1]))
            ranked.append((*chosen,max(c[0] for c in choices)))
        return sorted(ranked,key=lambda c:(-c[0],c[2]))


def paired_order(rows, settings, scores, guard):
    """Seed from reciprocal local neighbours before assigning isolated observations."""
    from .body_identity import anchor, independent
    rows=sorted(rows,key=lambda r:(r['cue'],r['id']))
    candidates=[r for r in rows if anchor(r,settings)]
    nearest={}
    for i,a in enumerate(candidates):
        for b in candidates[i+1:i+9]:
            if a['cue']==b['cue']:
                continue
            score=scores.face_similarity(a['_vector'],b['_vector'])
            if score<max(settings['match_threshold'],settings['cohesion_threshold']):
                continue
            if not guard.allows(a,[b]) or not guard.allows(b,[a]):
                continue
            for x,y in ((a,b),(b,a)):
                key=(-score,y['id'])
                if x['id'] not in nearest or key<nearest[x['id']]:
                    nearest[x['id']]=key
    by_id={r['id']:r for r in candidates}
    paired=[]
    for oid,(negative,peer) in nearest.items():
        if oid<peer and nearest.get(peer,(None,None))[1]==oid:
            pair=[by_id[oid],by_id[peer]]
            if len(independent(pair,settings))==2:
                paired.append((negative,oid,peer))
    first=[by_id[oid] for _,a,b in sorted(paired) for oid in (a,b)]
    seeded={r['id'] for r in first}
    return first+[r for r in rows if r['id'] not in seeded]
