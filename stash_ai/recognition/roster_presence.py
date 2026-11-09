"""Independent scene presence measured on the VTT clock."""
from bisect import bisect_left


def entrant_score(scores, query, witnesses, settings, maximum_bonus):
    floor = settings['match_threshold'] - maximum_bonus - settings['ambiguity_margin']
    if witnesses:
        return max(floor,scores.compare([query],witnesses)[0])
    return floor


class Presence:
    def __init__(self, witnesses, shot_at):
        self.rows = sorted(witnesses, key=lambda r:(r['seconds'],r['id']))
        self.seconds = [r['seconds'] for r in self.rows]
        self.shots = shot_at
        self.diverse = len({shot_at[r['cue']] for r in self.rows}) > 1
        self.order = {}
        for cue, shot in sorted(shot_at.items()):
            self.order.setdefault(shot, len(self.order))

    def weight(self, row):
        pos = bisect_left(self.seconds, row['seconds'])
        nearby = self.rows[max(0,pos-1):pos+1]
        query_shot = self.order[self.shots[row['cue']]]
        distance = min(abs(row['seconds']-r['seconds'])/30 +
                       max(0,abs(query_shot-self.order[self.shots[r['cue']]])-1)/4 for r in nearby)
        strength = .5 + .25*self.diverse + .25*(0 < pos < len(self.rows))
        return strength/(1+distance), [r['id'] for r in nearby]
