"""Fetch every dataset the project needs into ./data.

Models are not handled here. Transformers downloads those itself on first use,
into ~/.cache/huggingface/hub, so there is nothing to place by hand.

Split sizes are checked against Table 5 of the MAPLE paper, so a mirror that has
quietly changed cannot pass silently.
"""
import io
import os
import sys
import urllib.request
import zipfile

from . import data

DATA_DIR = data.DATA_DIR

EXPECTED = {
    "xsum": (204045, 11334), "date": (369, 250), "salient": (998, 250),
    "tracking": (1750, 250), "fp": (2952, 500), "banking77": (10003, 3080),
    "goemo": (43410, 5427), "gpqa": (250, 198),
}

BIGBENCH = ("https://raw.githubusercontent.com/google/BIG-bench/main/"
            "bigbench/benchmark_tasks")
BIGBENCH_PATH = {
    "date": "date_understanding",
    "salient": "salient_translation_error_detection",
    "tracking": "tracking_shuffled_objects/seven_objects",
}
BBH_CONFIG = {
    "date": "date_understanding",
    "salient": "salient_translation_error_detection",
    "tracking": "tracking_shuffled_objects_seven_objects",
}
FP_ZIP = ("https://huggingface.co/datasets/takala/financial_phrasebank/"
          "resolve/main/data/FinancialPhraseBank-v1.0.zip")


def fetch_bbh():
    """BIG-Bench supplies the training pool; BBH supplies the 250-example test split."""
    import json

    from datasets import load_dataset
    for task, path in BIGBENCH_PATH.items():
        origin = os.path.join(DATA_DIR, f"{task}_origin.json")
        if not os.path.exists(origin):
            with urllib.request.urlopen(f"{BIGBENCH}/{path}/task.json") as response:
                open(origin, "wb").write(response.read())

        evaluation = os.path.join(DATA_DIR, f"{task}_eval.json")
        if not os.path.exists(evaluation):
            dataset = load_dataset("lukaemon/bbh", BBH_CONFIG[task])
            rows = [dict(r) for r in dataset[list(dataset.keys())[0]]]
            json.dump({"examples": rows}, open(evaluation, "w"))

        # data.load regenerates the transformed training file when it is absent.
        transformed = os.path.join(DATA_DIR, f"{task}_train.json")
        if os.path.exists(transformed):
            os.remove(transformed)
        report(task)


def fetch_fp():
    target = os.path.join(DATA_DIR, "FinancialPhraseBank-v1.0", "Sentences_75Agree.txt")
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with urllib.request.urlopen(FP_ZIP) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in archive.namelist():
            if member.endswith("Sentences_75Agree.txt") and "__MACOSX" not in member:
                open(target, "wb").write(archive.read(member))
                break
    report("fp")


def fetch_huggingface():
    for task in ("banking77", "goemo", "xsum"):
        report(task)


def fetch_gpqa():
    """GPQA is gated; it needs a licence acceptance and a token."""
    main = os.path.join(DATA_DIR, "dataset", "gpqa_main.csv")
    if os.path.exists(main):
        report("gpqa")
        return
    os.makedirs(os.path.dirname(main), exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        for name in ("gpqa_main.csv", "gpqa_diamond.csv"):
            source = hf_hub_download("Idavidrein/gpqa", name, repo_type="dataset")
            open(os.path.join(DATA_DIR, "dataset", name), "wb").write(open(source, "rb").read())
        report("gpqa")
    except Exception as error:
        print(f"SKIP gpqa       gated dataset ({type(error).__name__})")
        print("     Accept at https://huggingface.co/datasets/Idavidrein/gpqa,")
        print("     run `hf auth login`, then re-run this script.")


def report(task):
    try:
        train, test = data.load(task)
    except Exception as error:
        print(f"FAIL {task:10s} {type(error).__name__}: {str(error)[:60]}")
        return False
    expected = EXPECTED[task]
    ok = (len(train), len(test)) == expected
    print(f"{'OK  ' if ok else 'WARN'} {task:10s} train={len(train):>7,} test={len(test):>6,}"
          f"   (paper: {expected[0]:,} / {expected[1]:,})")
    return ok


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    groups = sys.argv[1:] or ["bbh", "fp", "hf", "gpqa"]
    print("Fetching datasets, verified against Table 5 of the paper\n")
    if "bbh" in groups:
        fetch_bbh()
    if "fp" in groups:
        fetch_fp()
    if "hf" in groups:
        fetch_huggingface()
    if "gpqa" in groups:
        fetch_gpqa()


if __name__ == "__main__":
    main()
