"""Contract every reliability signal implements.

A signal reads a pseudo-labeled example and returns a score in [0, 1], where
higher means more trustworthy. Signals never see which backend produced the
label; they only call `ctx.model`.
"""
from dataclasses import dataclass, field


@dataclass
class Context:
    """Everything a signal may need, assembled once per run."""
    task: str
    model: object
    labeled: list = field(default_factory=list)   # human-labeled demos
    graph: object = None                          # MAPLE's train-pool graph
    pool_idx: object = None                       # node ids of the pseudo-labeled samples
    labeled_idx: object = None
    extra: dict = field(default_factory=dict)


class Signal:
    name = "signal"

    def score(self, item, ctx) -> float:
        """item is {"example": ..., "pred": ...}. Return a reliability score in [0, 1]."""
        raise NotImplementedError

    def score_all(self, items, ctx):
        return [self.score(it, ctx) for it in items]
