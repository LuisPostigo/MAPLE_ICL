"""Reliability gate, sitting between pseudo-labeling and demonstration selection.

Adding a signal means writing one class with a score or score_all method and
listing it in SIGNALS.
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
               weights=None, threshold=0.5, signal_options=None):
    signal_options = signal_options or {}
    signals = [SIGNALS[name](**signal_options.get(name, {})) for name in names]
    return Gate(signals, weights=weights, threshold=threshold)


__all__ = ["Context", "Signal", "Gate", "Decision", "weighted_mean",
           "ConfidenceSignal", "ConsistencySignal", "InfluenceSignal",
           "SIGNALS", "build_gate"]
