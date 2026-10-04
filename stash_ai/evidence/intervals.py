"""Ordered interval accumulation for retained evidence."""


def append_interval(intervals, start, end):
    if intervals and start <= intervals[-1][1]:
        intervals[-1][1] = max(intervals[-1][1], end)
    else:
        intervals.append([start, end])
