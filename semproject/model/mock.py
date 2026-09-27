"""Deterministic offline backend, so the whole pipeline runs with no model and no key."""
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
        h = int(hashlib.sha256(f"{prompt}{temperature:.3f}".encode()).hexdigest(), 16)
        labels = self.labels or ["(A)", "(B)", "(C)", "(D)"]
        text = labels[h % len(labels)]
        # A plausible logprob spread so the confidence signal has something to read.
        lp = -0.05 - (h % 100) / 100.0
        return Response(text=text, logprobs=[lp] * max(1, len(text.split())),
                        meta={"backend": "mock"})
