"""Datasets, prompts and scoring.

Prompt construction and scoring are reused from the upstream `process_*.py`
modules rather than reimplemented, so the replication stays faithful. Upstream's
`get_task.py` is bypassed because it fails at import (it pulls in three prompt
builders that were never published) and because three of its dataset ids were
retired when `datasets` dropped script-based loading.
"""
import json
import os
import re

import process_bank77 as B
import process_bbh as H
import process_fp as F
import process_goemotion as G
import process_gpqa as Q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BBH = {"date": "date", "salient": "salient", "tracking": "tracking"}
# Test splits that are never subsampled, so items align across seeds and methods.
FIXED_TEST = {"date", "salient", "tracking", "gpqa"}


def load(task):
    """Return (trainset, testset) as lists of dicts."""
    if task in BBH:
        train_p = os.path.join(DATA, f"{task}_train.json")
        if not os.path.exists(train_p):
            H.transform_data(json.load(open(os.path.join(DATA, f"{task}_origin.json"))), task)
        return (json.load(open(train_p)),
                json.load(open(os.path.join(DATA, f"{task}_eval.json")))["examples"])
    if task == "fp":
        return F.load_fp_data(os.path.join(DATA, "FinancialPhraseBank-v1.0",
                                           "Sentences_75Agree.txt"))
    if task == "gpqa":
        import pandas as pd
        d = pd.read_csv(os.path.join(DATA, "dataset", "gpqa_diamond.csv"))
        m = pd.read_csv(os.path.join(DATA, "dataset", "gpqa_main.csv"))
        return (Q.load_gpqa_examples(m[~m["Record ID"].isin(d["Record ID"])]),
                Q.load_gpqa_examples(d))

    from datasets import load_dataset
    repo, kw = {
        "banking77": ("legacy-datasets/banking77", {}),
        "goemo": ("google-research-datasets/go_emotions", {"name": "simplified"}),
        "xsum": ("EdinburghNLP/xsum", {}),
    }[task]
    ds = load_dataset(repo, cache_dir=DATA, **kw)
    return [dict(r) for r in ds["train"]], [dict(r) for r in ds["test"]]


def prompt(task, labeled, pseudo, query):
    """ICL prompt: labeled demos, then pseudo-labeled demos, then the query."""
    if task in BBH:
        return H.format_pseudobbh_prompt(labeled, pseudo, query)
    if task == "gpqa":
        return Q.format_pseudogpqa_prompt(labeled, pseudo, query)
    if task == "banking77":
        return B.format_pseudobanking77_prompt(labeled, pseudo, query)
    if task == "goemo":
        return G.format_pseudogoemo_prompt(labeled, pseudo, query)
    if task == "fp":
        return F.format_pseudofp_prompt(labeled, pseudo, query)
    head = "You are an expert in article summarization. Here are several examples.\n"
    for t in labeled:
        head += f"Article: {t['document']} \nSummary: {t['summary']}\n"
    for t in pseudo:
        head += f"Article: {t['example']['document']} \nSummary: {t['pred']}\n"
    return head + ("Give only the summary, and no extra commentary.\nArticle: "
                   + query["document"])


OPTION = re.compile(r"\(([A-Ga-g])\)")


def score_option(text, example):
    """Exact option-letter match. Replaces upstream's scorer for the choice tasks.

    Upstream normalises answers with a SQuAD routine that strips the article "a",
    so "(A)" becomes the empty string. That inverts scoring on every gold-(A)
    item: the correct answer trips the empty-response guard and scores 0, while
    any wrong answer contains the empty string and scores 1. Verified directly:

        gold (A), predicted (A)  -> upstream 0   (should be 1)
        gold (A), predicted (B)  -> upstream 1   (should be 0)

    Date is hit hardest because its training pool is 98.6% gold (A), which makes
    pseudo-label accuracy there meaningless under the upstream scorer.
    """
    gold = OPTION.findall((example.get("target") or "").strip())
    if not gold:
        return 0.0
    got = OPTION.findall(text or "")
    if not got:
        got = re.findall(r"\b([A-Ga-g])\b", text or "")
    return float(bool(got) and got[-1].upper() == gold[-1].upper())


def score(task, text, example, faithful=False):
    """Task metric for one prediction. 1/0 for the label tasks, ROUGE-L for XSum.

    `faithful=True` uses upstream's scorer verbatim, for reproducing the paper's
    published numbers. The default corrects the choice-task bug described above.
    """
    if task in BBH or task == "gpqa":
        if not faithful:
            return score_option(text, example)
        return float(H.calculate_bhh_acc(text, example) if task in BBH
                     else Q.calculate_gpqa_acc(text, example))
    if task == "banking77":
        return float(B.calculate_banking77_acc(text, example))
    if task == "goemo":
        return float(G.calculate_goemo_acc(text, example))
    if task == "fp":
        return float(F.calculate_fp_acc(text, example))
    if not text or not text.strip():
        return 0.0
    from rouge import Rouge
    try:
        return float(Rouge().get_scores(text, example["summary"])[0]["rouge-l"]["f"])
    except ValueError:
        return 0.0


def label_space(task):
    """Valid labels, where the task has a finite set. None for generation tasks."""
    if task in BBH or task == "gpqa":
        return ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
    if task == "banking77":
        return list(B.all_labels)
    if task == "goemo":
        return list(G.all_labels)
    if task == "fp":
        return ["positive", "negative", "neutral"]
    return None


def embed_text(task, example, with_label=False):
    """Retrieval text for one example; labels are included only for pool-side graphs."""
    if task in BBH or task == "gpqa":
        return f"Question: {example['input']}\nAnswer: " + (example["target"] if with_label else "")
    if task == "banking77":
        return f"service query: {example['text']}\nintent category: " + (
            B.all_labels[example["label"]] if with_label else "")
    if task == "goemo":
        return f"comment: {example['text']}\nemotion category: "
    if task == "fp":
        return f"Sentence: {example['sentence']}\nAnswer: " + (
            example["label"].strip() if with_label else "")
    return f"Article: {example['document']}\nSummary: " + (
        example["summary"] if with_label else "")
