"""Owner-thread frame progress and inexpensive, throttled buffer status rows."""
from contextlib import ExitStack


def ratio(value):
    return 'off' if value is None else f'{value[0]}/{value[1]}'


def memory_size(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if value < 1024 or unit == 'GiB':
            return f'{value:.1f}{unit}'
        value /= 1024


def buffer_lines(snapshot):
    current, following = snapshot['current'], snapshot['next']
    queued = f"Buffers: in={ratio(current['input'])} out={ratio(current['result'])}"
    if following is None:
        queued += ' | next=off'
    else:
        queued += (f" | next[{following['scene']}]: in={ratio(following['input'])}"
                   f" out={ratio(following['result'])}")
    used, limit = snapshot['cache']
    cache = f'{memory_size(used)}/{memory_size(limit)}' if limit else 'off'
    retained = f"Cache: {cache} | checkpoint interval={ratio(snapshot['checkpoint'])} batch"
    return queued, retained


class InferenceProgress:
    def __init__(self, factory, *, total, initial, desc, sample):
        self.factory, self.total, self.initial, self.desc = factory, total, initial, desc
        self.sample = sample
        self.stack = ExitStack()
        self.rows = []

    def __enter__(self):
        try:
            self.frame = self.stack.enter_context(self.factory(
                total=self.total, initial=self.initial, desc=self.desc, unit='frame', position=1,
                leave=False, dynamic_ncols=True, mininterval=.25, disable=None))
            if not self.frame.disable:
                self.rows = [self.stack.enter_context(self.factory(
                    total=0, desc='', bar_format='{desc}', position=position,
                    leave=False, dynamic_ncols=True, disable=None)) for position in (2, 3)]
                self.refresh()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *error):
        return self.stack.__exit__(*error)

    def update(self, count):
        # tqdm returns True only when it actually redraws the frame bar.
        if self.frame.update(count):
            self.refresh()

    def refresh(self):
        if self.rows:
            for row, text in zip(self.rows, buffer_lines(self.sample())):
                row.set_description_str(text)
