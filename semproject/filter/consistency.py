"""Signal 2 of 3: whether the label survives being asked again."""
from .base import Signal


class ConsistencySignal(Signal):
    name = "consistency"

    def __init__(self, samples=3, temperature=0.7, prompt_builder=None):
        self.samples = samples
        self.temperature = temperature
        self.prompt_builder = prompt_builder

    def score(self, item, context):
        """Fraction of resamples that reproduce the original pseudo-label.

        A confidently wrong model is consistently wrong, so this can endorse
        errors. Two variants worth testing later: perturbing the prompt instead
        of resampling, and cross-model agreement, which is the separate box in
        the design and needs only a second backend from the registry.
        """
        build_prompt = self.prompt_builder
        if build_prompt is None:
            from .. import data
            build_prompt = data.prompt

        prompt = build_prompt(context.task, context.labeled_demos, [], item["example"])
        original = (item["pred"] or "").strip().lower()

        matches = 0
        for _ in range(self.samples):
            resampled = context.model.generate(prompt, temperature=self.temperature).text
            matches += (resampled or "").strip().lower() == original
        return matches / self.samples
