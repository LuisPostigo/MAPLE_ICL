"""The only thing the rest of the codebase knows about a language model."""
from dataclasses import dataclass, field


@dataclass
class Response:
    """`logprobs` is None for backends that cannot report them, such as APIs."""
    text: str
    logprobs: list | None = None
    meta: dict = field(default_factory=dict)


class Model:
    name = "base"
    supports_logprobs = False

    def generate(self, prompt, temperature=0.0, max_tokens=64) -> Response:
        raise NotImplementedError

    def generate_many(self, prompts, **kwargs):
        return [self.generate(prompt, **kwargs) for prompt in prompts]
