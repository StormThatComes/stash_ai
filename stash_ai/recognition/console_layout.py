"""Recognition terminal formatting shared by the run and report commands."""
from collections import deque


def clock(seconds):
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def duration(seconds):
    return f'{max(0, seconds)*1000:.0f} ms' if seconds < 1 else clock(seconds)


def time_line(elapsed, phases):
    slowest = sorted(phases, key=phases.get, reverse=True)[:3]
    suffix = ', '.join(f'{name} {duration(phases[name])}' for name in slowest)
    return 'total=' + duration(elapsed) + (' | longest: ' + suffix if suffix else '')


def compact(lines, width):
    return [line if len(line) <= width else line[:max(0, width-1)] + '…' for line in lines]


class WindowRate:
    """Only bounded samples of fresh progress contribute to a phase estimate."""
    def __init__(self, started, initial):
        self.samples = deque([(started, initial)], maxlen=128)

    def sample(self, now, completed):
        samples = self.samples
        if now - samples[-1][0] > 30:
            samples.clear()
        if not samples or now > samples[-1][0]:
            samples.append((now, completed))
        while len(samples) > 1 and now - samples[0][0] > 30:
            samples.popleft()
        seconds = now - samples[0][0]
        delta = completed - samples[0][1]
        if seconds >= 1 and delta > 0:
            return delta / seconds
        return None
