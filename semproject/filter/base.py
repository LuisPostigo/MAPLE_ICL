"""Contract every reliability signal implements."""
from dataclasses import dataclass, field


@dataclass
class Context:
    """Everything a signal may need, assembled once per run."""
    task: str
    model: object
    labeled_demos: list = field(default_factory=list)
    graph: object = None
    pool_indices: object = None
    labeled_indices: object = None
    extra: dict = field(default_factory=dict)


class Signal:
    name = "signal"

    def score(self, item, context) -> float:
        """Reliability of one pseudo-labeled item in [0, 1], higher being safer.

        `item` is {"example": ..., "pred": ..., "logprobs": ...}.
        """
        raise NotImplementedError

    def score_all(self, items, context):
        return [self.score(item, context) for item in items]
