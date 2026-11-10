"""Independent scene presence measured in sampled cues and shot boundaries."""
from bisect import bisect_left, bisect_right


def entrant_score(scores, query, witnesses, settings, maximum_bonus):
    floor = settings['match_threshold'] - maximum_bonus - settings['ambiguity_margin']
    if witnesses:
        return max(floor,scores.compare([query],witnesses)[0])
    return floor


class Presence:
    def __init__(self, witnesses, shot_at, cut_starts=()):
        self.rows = sorted(witnesses, key=lambda r:(r['cue'],r['id']))
        self.cues = [r['cue'] for r in self.rows]
        self.cut_starts = cut_starts
        self.shots = shot_at
        self.diverse = len({shot_at[r['cue']] for r in self.rows}) > 1
        self.order = {}
        for cue, shot in sorted(shot_at.items()):
            self.order.setdefault(shot, len(self.order))

    def weight(self, row):
        pos = bisect_left(self.cues, row['cue'])
        nearby = self.rows[max(0,pos-1):pos+1]
        def shot_number(cue):
            return bisect_right(self.cut_starts,cue) if self.cut_starts else self.order[self.shots[cue]]
        query_shot = shot_number(row['cue'])
        distance = min(abs(row['cue']-r['cue'])/6 +
                       max(0,abs(query_shot-shot_number(r['cue']))-1)/4 for r in nearby)
        strength = .5 + .25*self.diverse + .25*(0 < pos < len(self.rows))
        return strength/(1+distance), [r['id'] for r in nearby]


class EntrantPaths:
    """An anonymous path after a cut may challenge a borderline roster claim."""
    def __init__(self, rows, anonymous, settings, cut_starts):
        from collections import defaultdict
        from .body_identity import independent, quality
        groups = defaultdict(list)
        self.paths = {}
        self.settings = settings
        for row in rows:
            key = anonymous.get(row['id'])
            if key and not row.get('face_person') and not quality(row).get('query_only'):
                groups[key].append(row)
        by_cue = defaultdict(list)
        for row in rows:
            by_cue[row['cue']].append(row)
        for group in groups.values():
            first = min(row['cue'] for row in group)
            shot = quality(group[0]).get('view_shot')
            if not isinstance(shot,int):
                continue
            after_cut = bool(cut_starts and bisect_right(cut_starts,first)>=2 and 0<=first-shot<=2)
            previous = by_cue.get(first-1, ())
            current = by_cue[first]
            partners = {r.get('face_person') for r in current} & {r.get('face_person') for r in previous} - {None}
            birth = len(current)>len(previous) or bool(partners)
            if not after_cut and not (birth and len({r['cue'] for r in group if quality(r).get('view_shot')==shot})>=3):
                continue
            votes = independent(group,settings)
            if len(votes)>=3:
                # Bounded, evenly spaced, independent challenger witnesses.
                if len(votes)>5:
                    votes = [votes[i*(len(votes)-1)//4] for i in range(5)]
                for row in group:
                    self.paths[row['id']] = votes

    def challenge(self, row, witnesses, scores, prior):
        from .body_identity import quality
        digest = quality(row).get('pixel_digest')
        peers = [p for p in self.paths.get(row['id'],()) if p['cue']!=row['cue'] and
                 (not digest or quality(p).get('pixel_digest')!=digest)]
        if len(peers)<2:
            return prior
        # Two independent peers must disagree with the returning owner's bank.
        # The query never supplies evidence for its own alternative.
        if sum(scores.compare([p],witnesses)[0]<self.settings['match_threshold'] for p in peers)>=2:
            return max(prior,scores.compare([row],peers)[0])
        return prior
