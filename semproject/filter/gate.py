"""The gate: run every signal, combine into one score, decide keep or filter.

This is box 4 of the design. Signals produce f_1..f_n, the combiner G maps them
to r(x, y) in [0, 1], and a threshold turns r into a decision:

    r >= tau  ->  keep in the reliable demonstration pool
    r <  tau  ->  filter out, or route to re-labeling

`weighted_mean` is a deliberate placeholder for G. The plan is to replace it
with a learned combiner (logistic regression over the signals, fit against
revealed ground-truth correctness) once the signals have been validated as
detectors. Swapping it means passing a different `combine=` callable; nothing
else in the pipeline changes.
"""
from dataclasses import dataclass, field


@dataclass
class Decision:
    index: int
    signals: dict = field(default_factory=dict)
    r: float = 0.0
    keep: bool = True


def weighted_mean(signals, weights):
    """Placeholder G(.): weighted average of the signal scores, clipped to [0, 1]."""
    num = sum(signals[k] * weights.get(k, 1.0) for k in signals)
    den = sum(weights.get(k, 1.0) for k in signals) or 1.0
    return max(0.0, min(1.0, num / den))


class Gate:
    def __init__(self, signals, weights=None, threshold=0.5, combine=weighted_mean):
        self.signals = list(signals)
        self.weights = weights or {s.name: 1.0 for s in self.signals}
        self.threshold = threshold
        self.combine = combine

    def evaluate(self, items, ctx):
        """Score every pseudo-labeled item. Returns one Decision per item."""
        cols = {s.name: s.score_all(items, ctx) for s in self.signals}
        out = []
        for i in range(len(items)):
            sig = {name: float(vals[i]) for name, vals in cols.items()}
            r = self.combine(sig, self.weights)
            out.append(Decision(index=i, signals=sig, r=r, keep=r >= self.threshold))
        return out

    def apply(self, items, ctx):
        """Split the pool into what survives the gate and what does not."""
        d = self.evaluate(items, ctx)
        return ([items[x.index] for x in d if x.keep],
                [items[x.index] for x in d if not x.keep], d)
