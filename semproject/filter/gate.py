"""The gate: run every signal, combine into one score, decide keep or filter."""
from dataclasses import dataclass, field


@dataclass
class Decision:
    index: int
    signals: dict = field(default_factory=dict)
    reliability: float = 0.0
    keep: bool = True


def weighted_mean(signal_scores, weights):
    """PLACEHOLDER combiner.

    Stands in for a learned model, most likely a logistic regression fit against
    revealed pseudo-label correctness once the signals are validated as
    detectors. Swapping it means passing a different `combine=`; nothing else in
    the pipeline changes.
    """
    total = sum(signal_scores[name] * weights.get(name, 1.0) for name in signal_scores)
    divisor = sum(weights.get(name, 1.0) for name in signal_scores) or 1.0
    return max(0.0, min(1.0, total / divisor))


class Gate:
    def __init__(self, signals, weights=None, threshold=0.5, combine=weighted_mean):
        self.signals = list(signals)
        self.weights = weights or {signal.name: 1.0 for signal in self.signals}
        self.threshold = threshold
        self.combine = combine

    def evaluate(self, items, context):
        """One Decision per pseudo-labeled item."""
        scores_by_signal = {signal.name: signal.score_all(items, context)
                            for signal in self.signals}
        decisions = []
        for index in range(len(items)):
            signal_scores = {name: float(values[index])
                             for name, values in scores_by_signal.items()}
            reliability = self.combine(signal_scores, self.weights)
            decisions.append(Decision(index=index, signals=signal_scores,
                                      reliability=reliability,
                                      keep=reliability >= self.threshold))
        return decisions

    def apply(self, items, context):
        """Split the pool into what survives the gate and what does not."""
        decisions = self.evaluate(items, context)
        kept = [items[d.index] for d in decisions if d.keep]
        filtered = [items[d.index] for d in decisions if not d.keep]
        return kept, filtered, decisions
