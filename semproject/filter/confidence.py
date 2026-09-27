"""Signal 1 of 3: how confident the model was in the label it produced."""
import json
import math
import re

from .base import Signal

SELF_REPORT_PROMPT = (
    'Answer the question, then rate your confidence.\n'
    'Reply with JSON only: {"answer": "<answer>", "confidence": <0.0-1.0>}\n\n')

CONFIDENCE_PATTERN = re.compile(r'"confidence"\s*:\s*([01](?:\.\d+)?)')
NEUTRAL = 0.5


class ConfidenceSignal(Signal):
    name = "confidence"

    def __init__(self, fallback_to_self_report=True):
        self.fallback_to_self_report = fallback_to_self_report

    def score(self, item, context):
        """Mean token logprob mapped through exp, or a self-reported score.

        Whether confidence tracks correctness at all is the open question this
        signal exists to answer, so it must be validated as a detector before
        being trusted as a filter.
        """
        logprobs = item.get("logprobs")
        if logprobs:
            return float(min(1.0, math.exp(sum(logprobs) / len(logprobs))))
        if not self.fallback_to_self_report:
            return NEUTRAL

        question = item["example"].get("input", str(item["example"]))[:2000]
        reply = context.model.generate(SELF_REPORT_PROMPT + question, max_tokens=96).text

        match = CONFIDENCE_PATTERN.search(reply or "")
        if match:
            return float(min(1.0, max(0.0, float(match.group(1)))))
        try:
            return float(json.loads(reply).get("confidence", NEUTRAL))
        except Exception:
            return NEUTRAL
