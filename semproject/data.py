"""Datasets, prompts and scoring.

Prompt building and scoring are reused from the upstream process_* modules
rather than reimplemented, so replication stays faithful. Upstream's get_task
is bypassed because it fails at import, and three of its dataset ids were
retired when `datasets` dropped script-based loading.
"""
import json
import os
import re

import process_bank77
import process_bbh
import process_fp
import process_goemotion
import process_gpqa

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

BBH_TASKS = {"date", "salient", "tracking"}

# Test splits that are never subsampled, so items line up across seeds and methods.
FIXED_TEST_SPLITS = {"date", "salient", "tracking", "gpqa"}

HUGGINGFACE_REPOS = {
    "banking77": ("legacy-datasets/banking77", {}),
    "goemo": ("google-research-datasets/go_emotions", {"name": "simplified"}),
    "xsum": ("EdinburghNLP/xsum", {}),
}

OPTION_LETTER = re.compile(r"\(([A-Ga-g])\)")


def load(task):
    """Return (train_examples, test_examples) as lists of dicts."""
    if task in BBH_TASKS:
        train_path = os.path.join(DATA_DIR, f"{task}_train.json")
        if not os.path.exists(train_path):
            source = json.load(open(os.path.join(DATA_DIR, f"{task}_origin.json")))
            process_bbh.transform_data(source, task)
        test_path = os.path.join(DATA_DIR, f"{task}_eval.json")
        return json.load(open(train_path)), json.load(open(test_path))["examples"]

    if task == "fp":
        return process_fp.load_fp_data(
            os.path.join(DATA_DIR, "FinancialPhraseBank-v1.0", "Sentences_75Agree.txt"))

    if task == "gpqa":
        import pandas as pd
        diamond = pd.read_csv(os.path.join(DATA_DIR, "dataset", "gpqa_diamond.csv"))
        everything = pd.read_csv(os.path.join(DATA_DIR, "dataset", "gpqa_main.csv"))
        held_out = everything[~everything["Record ID"].isin(diamond["Record ID"])]
        return (process_gpqa.load_gpqa_examples(held_out),
                process_gpqa.load_gpqa_examples(diamond))

    from datasets import load_dataset
    repo, options = HUGGINGFACE_REPOS[task]
    dataset = load_dataset(repo, cache_dir=DATA_DIR, **options)
    return ([dict(row) for row in dataset["train"]],
            [dict(row) for row in dataset["test"]])


def prompt(task, labeled_demos, pseudo_demos, query):
    """ICL prompt: labeled demos, then pseudo-labeled demos, then the query."""
    if task in BBH_TASKS:
        return process_bbh.format_pseudobbh_prompt(labeled_demos, pseudo_demos, query)
    if task == "gpqa":
        return process_gpqa.format_pseudogpqa_prompt(labeled_demos, pseudo_demos, query)
    if task == "banking77":
        return process_bank77.format_pseudobanking77_prompt(
            labeled_demos, pseudo_demos, query)
    if task == "goemo":
        return process_goemotion.format_pseudogoemo_prompt(
            labeled_demos, pseudo_demos, query)
    if task == "fp":
        return process_fp.format_pseudofp_prompt(labeled_demos, pseudo_demos, query)

    lines = ["You are an expert in article summarization. Here are several examples.\n"]
    for demo in labeled_demos:
        lines.append(f"Article: {demo['document']} \nSummary: {demo['summary']}\n")
    for demo in pseudo_demos:
        lines.append(f"Article: {demo['example']['document']} \nSummary: {demo['pred']}\n")
    lines.append("Give only the summary, and no extra commentary.\n")
    lines.append(f"Article: {query['document']}")
    return "".join(lines)


def score_option(text, example):
    """Exact option-letter match, replacing upstream's scorer for the choice tasks.

    Upstream normalises answers with a SQuAD routine that strips the article
    "a", so "(A)" becomes the empty string. That inverts scoring on every
    gold-(A) item: the correct answer trips the empty-response guard and scores
    0, while any wrong answer contains the empty string and scores 1. Date is
    hit hardest, since its training pool is 98.6 percent gold (A).
    """
    gold = OPTION_LETTER.findall((example.get("target") or "").strip())
    if not gold:
        return 0.0
    predicted = OPTION_LETTER.findall(text or "")
    if not predicted:
        predicted = re.findall(r"\b([A-Ga-g])\b", text or "")
    return float(bool(predicted) and predicted[-1].upper() == gold[-1].upper())


def score(task, text, example, faithful=False):
    """Task metric for one prediction, 1/0 for label tasks and ROUGE-L for XSum.

    `faithful=True` uses upstream's scorer verbatim, for reproducing published
    numbers. The default corrects the choice-task inversion described above.
    """
    if task in BBH_TASKS or task == "gpqa":
        if not faithful:
            return score_option(text, example)
        if task in BBH_TASKS:
            return float(process_bbh.calculate_bhh_acc(text, example))
        return float(process_gpqa.calculate_gpqa_acc(text, example))
    if task == "banking77":
        return float(process_bank77.calculate_banking77_acc(text, example))
    if task == "goemo":
        return float(process_goemotion.calculate_goemo_acc(text, example))
    if task == "fp":
        return float(process_fp.calculate_fp_acc(text, example))

    if not text or not text.strip():
        return 0.0
    from rouge import Rouge
    try:
        return float(Rouge().get_scores(text, example["summary"])[0]["rouge-l"]["f"])
    except ValueError:
        return 0.0


def label_space(task):
    """Valid labels where the task has a finite set, otherwise None."""
    if task in BBH_TASKS or task == "gpqa":
        return ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
    if task == "banking77":
        return list(process_bank77.all_labels)
    if task == "goemo":
        return list(process_goemotion.all_labels)
    if task == "fp":
        return ["positive", "negative", "neutral"]
    return None


def embed_text(task, example, with_label=False):
    """Retrieval text for one example; labels are included only for pool-side graphs."""
    if task in BBH_TASKS or task == "gpqa":
        answer = example["target"] if with_label else ""
        return f"Question: {example['input']}\nAnswer: {answer}"
    if task == "banking77":
        answer = process_bank77.all_labels[example["label"]] if with_label else ""
        return f"service query: {example['text']}\nintent category: {answer}"
    if task == "goemo":
        return f"comment: {example['text']}\nemotion category: "
    if task == "fp":
        answer = example["label"].strip() if with_label else ""
        return f"Sentence: {example['sentence']}\nAnswer: {answer}"
    answer = example["summary"] if with_label else ""
    return f"Article: {example['document']}\nSummary: {answer}"


OPTION_BLOCK = re.compile(r"\n\s*Options:\s*\n(?:\([A-G]\)[^\n]*\n?)+", re.MULTILINE)
OPTION_LINE = re.compile(r"\(([A-G])\)\s*([^\n]+)")


def gold_answer_text(example):
    """The text of the correct option, rather than its letter.

    Returns None when the example has no option block to read, which is every
    task outside the multiple-choice group.
    """
    options = dict(OPTION_LINE.findall(example.get("input", "")))
    letter = OPTION_LETTER.findall((example.get("target") or "").strip())
    if not letter or letter[0].upper() not in options:
        return None
    return options[letter[0].upper()].strip().rstrip(".")


def strip_options(text):
    return OPTION_BLOCK.sub("\n", text).rstrip()


def open_prompt(task, labeled_demos, pseudo_demos, query):
    """Same task with the answer options hidden, so the label must be generated.

    Only worth using where the answer space is genuinely large. Across the
    training pools there are 322 distinct gold answers for date and 47 for
    tracking, but just 6 for salient, which stays a closed set either way.
    """
    lines = ["You are an expert at answering questions. Here are several examples.\n"]
    for demo in labeled_demos:
        lines.append(f"Question: {strip_options(demo['input'])}\n"
                     f"Answer: {gold_answer_text(demo)}\n")
    for demo in pseudo_demos:
        lines.append(f"Question: {strip_options(demo['example']['input'])}\n"
                     f"Answer: {demo['pred']}\n")
    lines.append("Give only the answer itself, with no options, commentary or "
                 "explanation.\n")
    lines.append(f"Question: {strip_options(query['input'])}\nAnswer: ")
    return "".join(lines)


def normalise_answer(text):
    return " ".join((text or "").lower().replace(".", " ").split())


def score_open(text, example):
    """Match a generated answer against the gold option text.

    Accepts the gold answer appearing anywhere in the reply, since a model that
    reasons aloud still ends on the right string.
    """
    gold = gold_answer_text(example)
    if gold is None:
        return 0.0
    gold_norm = normalise_answer(gold)
    return float(bool(gold_norm) and gold_norm in normalise_answer(text))
