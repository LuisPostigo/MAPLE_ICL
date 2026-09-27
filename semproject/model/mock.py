"""Deterministic offline backend, so the pipeline runs with no weights and no key."""
import hashlib

from .base import Model, Response


class MockModel(Model):
    name = "mock"
    supports_logprobs = True

    def __init__(self, labels=None):
        self.labels = labels

    def set_labels(self, labels):
        self.labels = labels

    def generate(self, prompt, temperature=0.0, max_tokens=64):
        digest = int(hashlib.sha256(
            f"{prompt}{temperature:.3f}".encode()).hexdigest(), 16)
        labels = self.labels or ["(A)", "(B)", "(C)", "(D)"]
        answer = labels[digest % len(labels)]
        token_logprob = -0.05 - (digest % 100) / 100.0
        return Response(text=answer,
                        logprobs=[token_logprob] * max(1, len(answer.split())),
                        meta={"backend": "mock"})
