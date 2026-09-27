"""Signal 2 of 3: consistency.

v0 re-queries the same example k times at non-zero temperature and reports the
fraction of samples that reproduce the original pseudo-label. Stability under
resampling is a cheap proxy for the model not guessing.

Two variants worth testing later, both sketched here rather than implemented:
  - prompt perturbation: reword or reorder the demos instead of resampling
  - cross-model agreement: the separate "model agreement" box in the diagram,
    which needs a second backend and is a drop-in given the model registry

Known weakness: a confidently wrong model is consistently wrong, so this signal
can endorse errors. That is exactly what the detection experiment measures.
"""
from .base import Signal


class ConsistencySignal(Signal):
    name = "consistency"

    def __init__(self, k=3, temperature=0.7, prompt_fn=None):
        self.k = k
        self.temperature = temperature
        self.prompt_fn = prompt_fn      # (task, labeled, [], example) -> prompt

    def score(self, item, ctx):
        from .. import data
        build = self.prompt_fn or data.prompt
        p = build(ctx.task, ctx.labeled, [], item["example"])
        base = (item["pred"] or "").strip().lower()
        hits = 0
        for _ in range(self.k):
            r = ctx.model.generate(p, temperature=self.temperature).text
            hits += (r or "").strip().lower() == base
        return hits / self.k
