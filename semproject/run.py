"""End-to-end run: MAPLE, plus the reliability gate.

Follows the six boxes of the design: data, influence-based selection,
pseudo-labeling, the reliability gate, the surviving demo pool, then
query-adaptive selection and inference.

Task 1 is this with --no-gate, which reproduces MAPLE. Task 2 adds the gate and
reports whether its score separates correct pseudo-labels from wrong ones.
"""
import argparse
import json
import random

import numpy as np
import torch

from . import data, maple
from .filter import Context, build_gate
from .model import get_model


def detection_auc(scores, correctness):
    """Probability a signal ranks a correct pseudo-label above a wrong one.

    Ties count half. 0.5 is chance, so a signal at 0.5 cannot filter anything.
    Computed pairwise rather than by rank sum, because signals like consistency
    take few distinct values and tie handling is where rank sums go wrong.
    """
    reliable = [s for s, correct in zip(scores, correctness) if correct > 0.5]
    unreliable = [s for s, correct in zip(scores, correctness) if correct <= 0.5]
    if not reliable or not unreliable:
        return None
    wins = sum((good > bad) + 0.5 * (good == bad)
               for good in reliable for bad in unreliable)
    return wins / (len(reliable) * len(unreliable))


def parse_args():
    parser = argparse.ArgumentParser(description="MAPLE + reliability gate")
    parser.add_argument("-t", "--task", default="date")
    parser.add_argument("-m", "--model", default="mock",
                        help="backend name; see semproject.model.available()")
    parser.add_argument("--model-id", default=None, help="backend-specific model id")
    parser.add_argument("-nl", "--labeled", type=int, default=20)
    parser.add_argument("-nu", "--pseudo", type=int, default=20)
    parser.add_argument("-d", "--degree", type=int, default=20)
    parser.add_argument("-a", "--alpha", type=float, default=0.75)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=1000, help="train pool size")
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--signals", nargs="+",
                        default=["confidence", "consistency", "influence"])
    parser.add_argument("--weights", default=None,
                        help='JSON, e.g. \'{"confidence": 2}\'')
    parser.add_argument("--no-gate", action="store_true", help="plain MAPLE (Task 1)")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def build_splits(task, seed, pool_size, limit_test):
    train_examples, test_examples = data.load(task)

    sampled = np.random.choice(len(train_examples),
                               min(pool_size, len(train_examples)), replace=False)
    train_examples = [train_examples[i] for i in sampled]

    if task not in data.FIXED_TEST_SPLITS:
        sampled_test = np.random.choice(len(test_examples),
                                        min(300, len(test_examples)), replace=False)
        test_examples = [test_examples[i] for i in sampled_test]
    if limit_test:
        test_examples = test_examples[:limit_test]
    return train_examples, test_examples


def pseudo_label(model, task, labeled_demos, train_examples, selected_indices):
    labeled_items = []
    for index in selected_indices:
        example = train_examples[index]
        reply = model.generate(data.prompt(task, labeled_demos, [], example))
        labeled_items.append({"example": example, "pred": reply.text,
                              "logprobs": reply.logprobs})
    return labeled_items


def predict_test_set(model, task, labeled_demos, pool_items, test_examples, alpha):
    demo_count = len(labeled_demos) + len(pool_items)

    labeled_texts = [data.embed_text(task, demo, True) for demo in labeled_demos]
    pool_texts = [data.embed_text(task, item["example"], True) for item in pool_items]
    demo_graph = maple.knn_graph(maple.embed(labeled_texts + pool_texts),
                                 min(10, max(2, demo_count - 1)))

    query_side_texts = ([data.embed_text(task, demo) for demo in labeled_demos]
                        + [data.embed_text(task, item["example"]) for item in pool_items])
    query_side_embeddings = maple.embed(query_side_texts)

    scores = []
    for query in test_examples:
        query_embedding = maple.embed(data.embed_text(task, query))
        keep = maple.adaptive_select(demo_graph, query_side_embeddings,
                                     query_embedding, demo_count, alpha)
        chosen_labeled = [labeled_demos[i] for i in keep if i < len(labeled_demos)]
        chosen_pseudo = [pool_items[i - len(labeled_demos)] for i in keep
                         if i >= len(labeled_demos)]
        reply = model.generate(data.prompt(task, chosen_labeled, chosen_pseudo, query))
        scores.append(data.score(task, reply.text, query))
    return scores


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args.model, **({"model_id": args.model_id} if args.model_id else {}))
    if args.model == "mock":
        model.set_labels(data.label_space(args.task) or ["summary"])

    train_examples, test_examples = build_splits(
        args.task, args.seed, args.pool, args.limit_test)

    labeled_indices = np.random.choice(len(train_examples), args.labeled, replace=False)
    labeled_demos = [train_examples[i] for i in labeled_indices]

    pool_texts = [data.embed_text(args.task, example) for example in train_examples]
    train_graph = maple.knn_graph(
        maple.embed(pool_texts, cache_tag=f"{args.task}_s{args.seed}_pool"), args.degree)
    influence = maple.influence_scores(train_graph, labeled_indices)
    selected_indices = maple.select_for_labeling(
        train_graph, labeled_indices, args.pseudo, scores=influence)

    labeled_items = pseudo_label(model, args.task, labeled_demos,
                                 train_examples, selected_indices)

    # Ground truth is withheld above and revealed only here, so scoring the
    # pseudo-labels costs no extra model calls.
    correctness = [data.score(args.task, item["pred"], item["example"])
                   for item in labeled_items]

    decisions = []
    pool_items = labeled_items
    if not args.no_gate:
        context = Context(
            task=args.task, model=model, labeled_demos=labeled_demos,
            graph=train_graph, pool_indices=selected_indices,
            labeled_indices=labeled_indices,
            extra={"graph_influence": [float(influence[i]) for i in selected_indices]})
        gate = build_gate(
            args.signals,
            weights=json.loads(args.weights) if args.weights else None,
            threshold=args.threshold)
        pool_items, _, decisions = gate.apply(labeled_items, context)

    scores = predict_test_set(model, args.task, labeled_demos, pool_items,
                              test_examples, args.alpha)

    result = {
        "task": args.task, "model": model.name, "seed": args.seed,
        "labeled": args.labeled, "pseudo": args.pseudo, "alpha": args.alpha,
        "gate": not args.no_gate, "n_test": len(test_examples),
        "accuracy": float(np.mean(scores)) if scores else 0.0,
        "pseudo_label_accuracy": float(np.mean(correctness)) if correctness else None,
        "pool_size": len(pool_items),
        "dropped": len(labeled_items) - len(pool_items),
    }
    if decisions:
        reliabilities = [decision.reliability for decision in decisions]
        result["gate_auc"] = detection_auc(reliabilities, correctness)
        result["signal_auc"] = {
            name: detection_auc([d.signals[name] for d in decisions], correctness)
            for name in decisions[0].signals}
        result["mean_reliability"] = float(np.mean(reliabilities))

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
