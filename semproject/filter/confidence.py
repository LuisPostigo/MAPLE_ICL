"""Signal 1 of 3: model confidence.

v0 uses the mean token logprob of the generated label, mapped through exp() into
[0, 1]. Backends that expose no logprobs fall back to asking the model to report
its own confidence as JSON, which is the weaker but universally available form.

Open question this signal exists to answer: is confidence actually correlated
with pseudo-label correctness? Language models are frequently confident and
wrong, so this must be validated as a detector before it is trusted as a filter.
"""
import json
import math
import re

from .base import Signal

ASK = ('Answer the question, then rate your confidence.\n'
       'Reply with JSON only: {"answer": "<answer>", "confidence": <0.0-1.0>}\n\n')


class ConfidenceSignal(Signal):
    name = "confidence"

    def __init__(self, fallback_json=True):
        self.fallback_json = fallback_json

    def score(self, item, ctx):
        lps = (item.get("logprobs") or None)
        if lps:
            return float(min(1.0, math.exp(sum(lps) / len(lps))))
        if not self.fallback_json:
            return 0.5
        raw = ctx.model.generate(ASK + item["example"].get("input",
                                 str(item["example"]))[:2000], max_tokens=96).text
        m = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', raw or "")
        if m:
            return float(min(1.0, max(0.0, float(m.group(1)))))
        try:
            return float(json.loads(raw).get("confidence", 0.5))
        except Exception:
            return 0.5
