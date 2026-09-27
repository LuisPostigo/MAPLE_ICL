"""Signal 3 of 3: how much the demonstration matters.  PLACEHOLDER."""
import numpy as np

from .base import Signal

NEUTRAL = 0.5


class InfluenceSignal(Signal):
    name = "influence"

    def score_all(self, items, context):
        """MAPLE's graph influence, rescaled to [0, 1] across the pool.

        PLACEHOLDER. The intended version measures how much a demonstration
        changes model behaviour, in the spirit of influence functions where the
        effect of upweighting a point scales with the learning rate. That needs
        gradient access and cannot run against an API backend.

        This stand-in is free: the graph exists before pseudo-labeling starts.
        Also free and worth testing alongside it, neighbourhood label agreement,
        meaning whether this label matches those of its graph neighbours.
        """
        raw_scores = context.extra.get("graph_influence")
        if raw_scores is None or len(raw_scores) != len(items):
            return [NEUTRAL] * len(items)

        scores = np.asarray(raw_scores, dtype=float)
        lowest, highest = float(scores.min()), float(scores.max())
        if highest - lowest < 1e-9:
            return [NEUTRAL] * len(items)
        return ((scores - lowest) / (highest - lowest)).tolist()

    def score(self, item, context):
        return NEUTRAL
