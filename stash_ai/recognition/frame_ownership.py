"""Owner-thread image leases shared by face microbatches and queued pose work."""
from stash_ai.runtime.cleanup import cleanup


class Lease:
    def __init__(self, image, owner=None):
        self.image = image
        self.owner = owner if owner is not None else [image, 0]
        self.owner[1] += 1

    def retain(self):
        if self.owner is None:
            raise RuntimeError('Cannot retain a released frame.')
        return Lease(self.image, self.owner)

    def close(self):
        owner, self.owner = self.owner, None
        if owner is not None:
            owner[1] -= 1
            if not owner[1]:
                owner[0].close()


class CueBatch(list):
    def __init__(self):
        super().__init__()
        self.leases = {}

    def add(self, index, cue, image):
        self.leases[id(image)] = Lease(image)
        self.append((index, cue, image))

    def retain(self, image):
        return self.leases[id(image)].retain()

    def close(self):
        leases, self.leases = self.leases, {}
        cleanup([lease.close for lease in leases.values()])
