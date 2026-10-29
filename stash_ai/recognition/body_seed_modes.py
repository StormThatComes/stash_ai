"""Frozen complete-link body appearances of an already known face identity."""


class FrozenSeeds:
    def __init__(self,seeds,scores,settings):
        self.seeds,self.scores,self.settings=seeds,scores,settings
        self.modes={}
        threshold=max(settings['match_threshold'],settings['cohesion_threshold'])
        for person,rows in seeds.items():
            modes=[]
            for row in rows:
                compatible=[m for m in modes if scores.compare([row],m)[1]>=threshold]
                if len(compatible)==1:compatible[0].append(row)
                else:modes.append([row])
            self.modes[person]=modes

    def rank(self,row,guard):
        from .body_identity import independent,copresent
        settings=self.settings;ranked=[]
        for person,modes in self.modes.items():
            allowed=not copresent([row],self.seeds[person]) and guard.allows(row,self.seeds[person],person)
            choices=[]
            for mode in modes:
                mean,minimum=self.scores.compare([row],mode)
                good=(allowed and mean>=settings['match_threshold'] and minimum>=settings['cohesion_threshold']
                      and len(independent(mode,settings))>=settings['attach_min_support'])
                choices.append((mean,minimum,person,mode,good))
            valid=[c for c in choices if c[4]]
            chosen=max(valid or choices,key=lambda c:(c[0],c[1]))
            ranked.append((*chosen,max(c[0] for c in choices)))
        return sorted(ranked,key=lambda c:(-c[0],c[2]))
