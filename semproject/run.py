"""End-to-end run: MAPLE, plus the reliability gate.

Follows the six boxes of the design.

  1 data            small labeled set D_L, large unlabeled pool D_U
  2 selection       MAPLE picks influential unlabeled samples
  3 pseudo-labeling the model labels them
  4 gate            reliability check r(x, y) in [0, 1]      <- this research
  5 pool            keep r >= tau, filter or re-label the rest
  6 inference       query-adaptive selection, then ICL

Task 1 is this with `--no-gate`, which reproduces MAPLE. Task 2 adds the gate and
reports whether its score actually separates correct pseudo-labels from wrong
ones, using ground truth that is withheld during labeling and revealed only for
scoring. That validation costs no extra model calls.
"""
import argparse
import json
import random

import numpy as np
import torch

from . import data, maple
from .filter import Context, build_gate
from .model import get_model


def auc(scores, labels):
    """P(signal ranks a correct pseudo-label above a wrong one), ties counting half.

    0.5 is chance, so a signal at 0.5 cannot be used as a filter. Computed as a
    direct pairwise comparison rather than a rank sum: signals like consistency
    take few distinct values, and tie handling is where rank-sum formulations
    silently go wrong.
    """
    pos = [s for s, y in zip(scores, labels) if y > 0.5]
    neg = [s for s, y in zip(scores, labels) if y <= 0.5]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser(description="MAPLE + reliability gate")
    p.add_argument("-t", "--task", default="date")
    p.add_argument("-m", "--model", default="mock",
                   help="backend name; see semproject.model.available()")
    p.add_argument("--model-id", default=None, help="backend-specific model id")
    p.add_argument("-nl", "--labeled", type=int, default=20)
    p.add_argument("-nu", "--pseudo", type=int, default=20)
    p.add_argument("-d", "--degree", type=int, default=20)
    p.add_argument("-a", "--alpha", type=float, default=0.75)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("--pool", type=int, default=1000, help="train pool size")
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--signals", nargs="+",
                   default=["confidence", "consistency", "influence"])
    p.add_argument("--weights", default=None, help='JSON, e.g. \'{"confidence":2}\'')
    p.add_argument("--no-gate", action="store_true", help="plain MAPLE (Task 1)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    kw = {"model_id": a.model_id} if a.model_id else {}
    model = get_model(a.model, **kw)
    if a.model == "mock":
        model.set_labels(data.label_space(a.task) or ["summary"])

    # 1 -- data
    train, test = data.load(a.task)
    idx = np.random.choice(len(train), min(a.pool, len(train)), replace=False)
    train = [train[i] for i in idx]
    if a.task not in data.FIXED_TEST:
        t = np.random.choice(len(test), min(300, len(test)), replace=False)
        test = [test[i] for i in t]
    if a.limit_test:
        test = test[:a.limit_test]
    lab_idx = np.random.choice(len(train), a.labeled, replace=False)
    labeled = [train[i] for i in lab_idx]

    # 2 -- influence-based selection over the train pool
    pool_txt = [data.embed_text(a.task, e) for e in train]
    G = maple.knn_graph(maple.embed(pool_txt, cache_tag=f"{a.task}_s{a.seed}_pool"), a.degree)
    inf_all = maple.influence_scores(G, lab_idx)
    sel = maple.select_for_labeling(G, lab_idx, a.pseudo, scores=inf_all)

    # 3 -- pseudo-labeling, using only the labeled demos as context
    items = []
    for i in sel:
        r = model.generate(data.prompt(a.task, labeled, [], train[i]))
        items.append({"example": train[i], "pred": r.text, "logprobs": r.logprobs})

    # ground truth, withheld above, revealed only to score the labels
    correct = [data.score(a.task, it["pred"], it["example"]) for it in items]

    # 4/5 -- the gate
    decisions, pool = [], items
    if not a.no_gate:
        ctx = Context(task=a.task, model=model, labeled=labeled, graph=G,
                      pool_idx=sel, labeled_idx=lab_idx,
                      extra={"graph_influence": [float(inf_all[i]) for i in sel]})
        gate = build_gate(a.signals, weights=json.loads(a.weights) if a.weights else None,
                          threshold=a.threshold)
        pool, dropped, decisions = gate.apply(items, ctx)

    # 6 -- query-adaptive selection, then ICL
    n_demos = len(labeled) + len(pool)
    pe = maple.embed([data.embed_text(a.task, e, True) for e in labeled]
                     + [data.embed_text(a.task, it["example"], True) for it in pool])
    Gp = maple.knn_graph(pe, min(10, max(2, n_demos - 1)))
    pe_q = maple.embed([data.embed_text(a.task, e) for e in labeled]
                       + [data.embed_text(a.task, it["example"]) for it in pool])
    scores = []
    for q in test:
        keep = maple.adaptive_select(Gp, pe_q, maple.embed(data.embed_text(a.task, q)),
                                     n_demos, a.alpha)
        ld = [labeled[k] for k in keep if k < len(labeled)]
        pd_ = [pool[k - len(labeled)] for k in keep if k >= len(labeled)]
        scores.append(data.score(a.task, model.generate(data.prompt(a.task, ld, pd_, q)).text, q))

    res = {
        "task": a.task, "model": model.name, "seed": a.seed,
        "labeled": a.labeled, "pseudo": a.pseudo, "alpha": a.alpha,
        "gate": not a.no_gate, "n_test": len(test),
        "accuracy": float(np.mean(scores)) if scores else 0.0,
        "pseudo_label_accuracy": float(np.mean(correct)) if correct else None,
        "pool_size": len(pool), "dropped": len(items) - len(pool),
    }
    if decisions:
        rs = [d.r for d in decisions]
        res["gate_auc_vs_truth"] = auc(rs, correct)
        res["signal_auc"] = {n: auc([d.signals[n] for d in decisions], correct)
                             for n in decisions[0].signals}
        res["mean_r"] = float(np.mean(rs))
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
