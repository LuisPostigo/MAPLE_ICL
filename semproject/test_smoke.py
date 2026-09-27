"""Cheap checks that the wiring holds. No model weights, no network."""
from . import data, maple
from .filter import Context, build_gate
from .model import get_model
from .run import detection_auc


def test_detection_auc_stays_in_range():
    assert detection_auc([.9, .8, .1, .2], [1, 1, 0, 0]) == 1.0
    assert detection_auc([.1, .2, .9, .8], [1, 1, 0, 0]) == 0.0
    assert detection_auc([.5] * 4, [1, 1, 0, 0]) == 0.5
    assert detection_auc([.5, .6], [1, 1]) is None
    print("OK  detection_auc in [0, 1] under ties")


def test_choice_scorer_is_corrected():
    example = {"target": "(A)", "input": "q"}
    assert data.score("date", "(A)", example) == 1.0
    assert data.score("date", "(B)", example) == 0.0
    assert data.score("date", "(A)", example, faithful=True) == 0.0
    print("OK  choice scorer corrected; faithful= still reproduces upstream")


def test_model_is_swappable():
    model = get_model("mock")
    model.set_labels(["(A)", "(B)"])
    reply = model.generate("hi")
    assert reply.text in {"(A)", "(B)"} and reply.logprobs
    print("OK  model reachable through one interface")


def test_gate_scores_and_splits():
    model = get_model("mock")
    model.set_labels(["positive", "negative", "neutral"])
    items = [{"example": {"sentence": f"s{i}", "label": "positive\n"},
              "pred": "positive", "logprobs": [-0.2]} for i in range(6)]
    context = Context(task="fp", model=model,
                      extra={"graph_influence": list(range(6))})
    kept, filtered, decisions = build_gate(("confidence", "influence"),
                                           threshold=0.4).apply(items, context)
    assert len(kept) + len(filtered) == len(items)
    assert all(0.0 <= d.reliability <= 1.0 for d in decisions)
    print(f"OK  gate ran: kept {len(kept)}, filtered {len(filtered)}")


def test_alpha_keeps_expected_fraction():
    embeddings = maple.embed([f"text number {i}" for i in range(24)])
    graph = maple.knn_graph(embeddings, 5)
    keep = maple.adaptive_select(graph, embeddings,
                                 maple.embed("text number 3"), 24, 0.75)
    assert len(keep) == 18
    print("OK  alpha=0.75 keeps 18 of 24")


if __name__ == "__main__":
    test_detection_auc_stays_in_range()
    test_choice_scorer_is_corrected()
    test_model_is_swappable()
    test_gate_scores_and_splits()
    test_alpha_keeps_expected_fraction()
    print("\nall smoke tests passed")
