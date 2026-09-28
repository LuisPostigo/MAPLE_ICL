"""Measure how good a model is at writing labels, before trusting it to write any.

A labeler that collapses onto one answer looks fine on accuracy alone when that
answer happens to be the majority class, so the collapse check matters as much
as the score.
"""
import argparse
import collections
import json

import numpy as np

from . import data
from .model import get_model


def measure(model, task, labeled_demos, held_out, build_prompt=None, score_fn=None):
    """Accuracy on examples whose real labels we already have.

    This is also the batch form of the consistency signal: rather than asking
    whether one answer is stable, it asks how often this model is right about
    this kind of question at all.
    """
    build_prompt = build_prompt or data.prompt
    score_fn = score_fn or (lambda text, example: data.score(task, text, example))

    predictions, scores = [], []
    for example in held_out:
        reply = model.generate(build_prompt(task, labeled_demos, [], example))
        predictions.append((reply.text or "").strip())
        scores.append(score_fn(reply.text, example))

    counts = collections.Counter(predictions)
    most_common, most_common_n = counts.most_common(1)[0] if counts else ("", 0)
    label_count = len(data.label_space(task) or [])

    return {
        "model": model.name,
        "task": task,
        "n": len(held_out),
        "accuracy": float(np.mean(scores)) if scores else 0.0,
        "chance": round(1.0 / label_count, 4) if label_count else None,
        "distinct_outputs": len(counts),
        "most_common_output": most_common[:40],
        "most_common_share": round(most_common_n / len(predictions), 3) if predictions else 0.0,
        "collapsed": bool(predictions) and most_common_n / len(predictions) > 0.8,
    }


def verdict(report):
    if report["collapsed"]:
        return "COLLAPSED -- answers almost everything the same way; unusable as a labeler"
    if report["chance"] and report["accuracy"] < report["chance"] * 2:
        return "TOO WEAK -- barely above guessing"
    if report["accuracy"] < 0.5:
        return "WEAK -- more than half its labels would be wrong"
    return "USABLE"


def main():
    parser = argparse.ArgumentParser(description="Check a model before using it to label")
    parser.add_argument("-t", "--task", default="banking77")
    parser.add_argument("-m", "--model", default="hf")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("-n", "--n-check", type=int, default=40)
    parser.add_argument("-nl", "--labeled", type=int, default=8)
    parser.add_argument("--open-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    model = get_model(args.model, **({"model_id": args.model_id} if args.model_id else {}))
    if args.model == "mock":
        model.set_labels(data.label_space(args.task) or ["summary"])

    train_examples, _ = data.load(args.task)
    sampled = np.random.choice(len(train_examples),
                               args.labeled + args.n_check, replace=False)
    chosen = [train_examples[i] for i in sampled]

    build_prompt = data.open_prompt if args.open_labels else None
    score_fn = data.score_open if args.open_labels else None

    report = measure(model, args.task, chosen[:args.labeled], chosen[args.labeled:],
                     build_prompt, score_fn)
    report["verdict"] = verdict(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
