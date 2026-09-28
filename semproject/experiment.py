"""The experiment that decides whether the reliability gate is worth having.

Four students are trained on the same task and compared. Only the training data
differs.

  human_only   the handful of real human labels, nothing else
  all_pseudo   human labels plus every generated label, unfiltered
  filtered     human labels plus the generated labels the gate kept
  oracle       human labels plus only the generated labels that are actually
               correct, using ground truth no real system would have
  random_drop  human labels plus the same COUNT the gate kept, chosen at random

The gate is worth having if `filtered` beats `random_drop`, not if it beats
`all_pseudo`. Dropping examples changes two things at once: the training set
gets cleaner and it gets smaller. `random_drop` holds the size change fixed so
the comparison isolates *which* examples were dropped.

That confound also breaks the obvious reading of `oracle`. A perfect filter on a
weak labeler throws away most of the data, so `oracle` can score below
`all_pseudo` purely on volume. Read it as a ceiling only when the labeler is
good enough that the surviving set is still large.
"""
import argparse
import json

import numpy as np

from . import data, maple
from .filter import Context, build_gate
from .model import get_model
from .train import evaluate_student, format_pair, train_student


def run_arm(name, task, pairs, test_examples, student_id, epochs, log):
    log(f"  {name}: training on {len(pairs)} examples")
    model, tokenizer, history = train_student(
        task, pairs, model_id=student_id, epochs=epochs, log=log)
    accuracy = evaluate_student(model, tokenizer, task, test_examples)
    log(f"  {name}: accuracy {accuracy:.4f}")
    del model
    return {"arm": name, "n_train": len(pairs), "accuracy": accuracy,
            "final_loss": history[-1]}


def main():
    parser = argparse.ArgumentParser(description="Does the gate improve the student?")
    parser.add_argument("-t", "--task", default="banking77")
    parser.add_argument("--labeler", default="hf", help="model that writes the labels")
    parser.add_argument("--labeler-id", default=None)
    parser.add_argument("--student-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("-nl", "--labeled", type=int, default=20)
    parser.add_argument("-nu", "--pseudo", type=int, default=100)
    parser.add_argument("--pool", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--degree", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--signals", nargs="+", default=["confidence", "consistency"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+",
                        default=["human_only", "all_pseudo", "filtered",
                                 "random_drop", "oracle"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log = print
    np.random.seed(args.seed)

    labeler = get_model(args.labeler,
                        **({"model_id": args.labeler_id} if args.labeler_id else {}))
    if args.labeler == "mock":
        labeler.set_labels(data.label_space(args.task) or ["summary"])

    train_examples, test_examples = data.load(args.task)
    pool = np.random.choice(len(train_examples), min(args.pool, len(train_examples)),
                            replace=False)
    train_examples = [train_examples[i] for i in pool]
    test_examples = [test_examples[i] for i in
                     np.random.choice(len(test_examples),
                                      min(args.n_test, len(test_examples)), replace=False)]

    labeled_indices = np.random.choice(len(train_examples), args.labeled, replace=False)
    labeled_demos = [train_examples[i] for i in labeled_indices]

    log("selecting which unlabeled examples to label")
    graph = maple.knn_graph(
        maple.embed([data.embed_text(args.task, e) for e in train_examples],
                    cache_tag=f"{args.task}_s{args.seed}_pool"), args.degree)
    influence = maple.influence_scores(graph, labeled_indices)
    selected = maple.select_for_labeling(graph, labeled_indices, args.pseudo,
                                         scores=influence)

    log(f"writing {len(selected)} labels with {labeler.name}")
    items = []
    for index in selected:
        example = train_examples[index]
        reply = labeler.generate(data.prompt(args.task, labeled_demos, [], example))
        items.append({"example": example, "pred": (reply.text or "").strip(),
                      "logprobs": reply.logprobs})

    correctness = [data.score(args.task, item["pred"], item["example"]) for item in items]
    label_accuracy = float(np.mean(correctness)) if correctness else 0.0
    log(f"  generated labels are {label_accuracy:.1%} correct")

    log("running the gate")
    context = Context(task=args.task, model=labeler, labeled_demos=labeled_demos,
                      graph=graph, pool_indices=selected,
                      labeled_indices=labeled_indices,
                      extra={"graph_influence": [float(influence[i]) for i in selected]})
    kept, dropped, decisions = build_gate(
        args.signals, threshold=args.threshold).apply(items, context)
    log(f"  gate kept {len(kept)}, dropped {len(dropped)}")

    human_pairs = [format_pair(args.task, e) for e in labeled_demos]
    pseudo_pairs = {
        "all_pseudo": [format_pair(args.task, i["example"], i["pred"]) for i in items],
        "filtered": [format_pair(args.task, i["example"], i["pred"]) for i in kept],
        "oracle": [format_pair(args.task, i["example"], i["pred"])
                   for i, ok in zip(items, correctness) if ok > 0.5],
    }
    # Same number of examples the gate kept, chosen at random rather than by score.
    survivors = np.random.choice(len(items), len(kept), replace=False)
    pseudo_pairs["random_drop"] = [
        format_pair(args.task, items[i]["example"], items[i]["pred"]) for i in survivors]

    results = []
    for arm in args.arms:
        pairs = human_pairs + (pseudo_pairs.get(arm, []) if arm != "human_only" else [])
        results.append(run_arm(arm, args.task, pairs, test_examples,
                               args.student_id, args.epochs, log))

    summary = {
        "task": args.task, "labeler": labeler.name, "student": args.student_id,
        "seed": args.seed, "n_human": args.labeled, "n_pseudo": len(items),
        "label_accuracy": label_accuracy,
        "gate_kept": len(kept), "gate_dropped": len(dropped),
        "headline": "gate helps only if filtered > random_drop",
        "n_test": len(test_examples), "arms": results,
    }
    print()
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
