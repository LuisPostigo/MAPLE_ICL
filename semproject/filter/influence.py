"""Signal 3 of 3: influence / usefulness.  PLACEHOLDER.

The intended version measures how much a demonstration changes the model's
behaviour, in the spirit of influence functions, where the effect of upweighting
a training point is scaled by the learning rate. That needs gradient access and
is not implementable against an API backend.

v0 substitutes the one influence measure that is already available for free:
MAPLE has embedded every sample and built the neighbour graph before pseudo-
labeling begins, so a sample's graph influence on the labeled set costs nothing
to read back. Normalised to [0, 1] across the pool.

Worth testing alongside it, also free: neighbourhood label agreement, meaning
whether this pseudo-label matches the labels of its graph neighbours. A label
that disagrees with everything around it is suspect, and computing that requires
no additional model calls at all.
"""
import numpy as np

from .base import Signal


class InfluenceSignal(Signal):
    name = "influence"

    def score_all(self, items, ctx):
        raw = ctx.extra.get("graph_influence")
        if raw is None or len(raw) != len(items):
            return [0.5] * len(items)           # neutral when unavailable
        v = np.asarray(raw, float)
        lo, hi = float(v.min()), float(v.max())
        return [0.5] * len(items) if hi - lo < 1e-9 else ((v - lo) / (hi - lo)).tolist()

    def score(self, item, ctx):
        return 0.5
