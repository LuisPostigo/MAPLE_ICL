"""Cheap checks that the wiring holds. No model weights, no network."""
import numpy as np

from . import data, maple
from .filter import Context, build_gate
from .model import get_model
from .run import auc


def test_auc_bounds():
    assert auc([.9, .8, .1, .2], [1, 1, 0, 0]) == 1.0
    assert auc([.1, .2, .9, .8], [1, 1, 0, 0]) == 0.0
    assert auc([.5] * 4, [1, 1, 0, 0]) == 0.5          # ties -> chance, not out of range
    assert auc([.5, .6], [1, 1]) is None
    print("OK  auc in [0,1] under ties")


def test_scorer_fix():
    ex = {"target": "(A)", "input": "q"}
    assert data.score("date", "(A)", ex) == 1.0
    assert data.score("date", "(B)", ex) == 0.0
    assert data.score("date", "(A)", ex, faithful=True) == 0.0   # upstream inverts
    print("OK  corrected scorer; faithful= still reproduces upstream")


def test_model_is_swappable():
    m = get_model("mock")
    m.set_labels(["(A)", "(B)"])
    r = m.generate("hi")
    assert r.text in {"(A)", "(B)"} and r.logprobs
    print("OK  model behind one interface")


def test_gate_runs():
    m = get_model("mock"); m.set_labels(["positive", "negative", "neutral"])
    items = [{"example": {"sentence": f"s{i}", "label": "positive\n"},
              "pred": "positive", "logprobs": [-0.2]} for i in range(6)]
    gate = build_gate(("confidence", "influence"), threshold=0.4)
    ctx = Context(task="fp", model=m, extra={"graph_influence": list(range(6))})
    kept, dropped, dec = gate.apply(items, ctx)
    assert len(kept) + len(dropped) == 6
    assert all(0.0 <= d.r <= 1.0 for d in dec)
    print(f"OK  gate ran: kept {len(kept)}, dropped {len(dropped)}")


def test_alpha():
    e = maple.embed([f"text number {i}" for i in range(24)])
    G = maple.knn_graph(e, 5)
    assert len(maple.adaptive_select(G, e, maple.embed("text number 3"), 24, 0.75)) == 18
    print("OK  alpha=0.75 keeps 18 of 24")


if __name__ == "__main__":
    test_auc_bounds(); test_scorer_fix(); test_model_is_swappable()
    test_gate_runs(); test_alpha()
    print("\nall smoke tests passed")
