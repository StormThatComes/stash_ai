"""At most one authorized scene waiting for owner-thread result processing."""
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelScene:
    scene: str
    paths: dict
    cues: list
    media: Any
    signature: str
    input_options: Any = None
    checkpoint: Any = None
    state: Any = None
    batches: Any = None
    prepared: Any = None

    def close(self):
        if self.batches is not None:
            self.batches.close()
        elif self.prepared is not None:
            self.prepared.close()
