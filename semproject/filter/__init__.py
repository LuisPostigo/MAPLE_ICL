"""Reliability gate: the research contribution sitting between pseudo-labeling
and demonstration selection.

Adding a fourth signal (cross-model agreement, neighbourhood label agreement)
means writing one class with a `score`/`score_all` method and listing it here.
"""
from .base import Context, Signal
from .confidence import ConfidenceSignal
from .consistency import ConsistencySignal
from .gate import Decision, Gate, weighted_mean
from .influence import InfluenceSignal

SIGNALS = {
    "confidence": ConfidenceSignal,
    "consistency": ConsistencySignal,
    "influence": InfluenceSignal,
}


def build_gate(names=("confidence", "consistency", "influence"),
               weights=None, threshold=0.5, **kw):
    return Gate([SIGNALS[n](**kw.get(n, {})) for n in names],
                weights=weights, threshold=threshold)


__all__ = ["Context", "Signal", "Gate", "Decision", "weighted_mean",
           "ConfidenceSignal", "ConsistencySignal", "InfluenceSignal",
           "SIGNALS", "build_gate"]
