"""The only thing the rest of the codebase knows about a model.

Swapping backends means writing one subclass. Nothing outside `model/` imports a
backend directly; everything goes through `get_model(name)`.
"""
from dataclasses import dataclass, field


@dataclass
class Response:
    text: str
    # Per-token logprobs of the generated tokens, when the backend exposes them.
    # The confidence signal degrades to a self-reported score when this is None.
    logprobs: list | None = None
    meta: dict = field(default_factory=dict)


class Model:
    name = "base"
    supports_logprobs = False

    def generate(self, prompt, temperature=0.0, max_tokens=64) -> Response:
        raise NotImplementedError

    def generate_many(self, prompts, **kw):
        return [self.generate(p, **kw) for p in prompts]
