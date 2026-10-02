# semproject — MAPLE + a pseudo-label reliability gate

Everything we add lives here. Upstream files are untouched; this package imports
their prompt builders and scorers rather than reimplementing them.

```
semproject/
  model/      the LLM, behind one interface — swap backends without touching anything else
  filter/     the reliability gate (the research contribution)
  data.py     datasets, prompts, scoring
  maple.py    embeddings, graph, influence, adaptive selection
  run.py      end-to-end driver
```

## Setup

### Models: nothing to download by hand

There is no link to fetch and no folder to place anything in. Transformers pulls
the weights itself the first time they are used, into
`~/.cache/huggingface/hub`, and reuses them afterwards.

| Model | Role | On disk | When it arrives |
|---|---|---|---|
| `facebook/contriever-msmarco` | embeddings for MAPLE's graph | 836 MB | first run of anything |
| `Qwen/Qwen2.5-0.5B-Instruct` | writes labels, and is the student | 953 MB | first run that needs a local model |

Swap either with `--model-id`, and the new weights download the same way:

```bash
python -m semproject.run --model-id Qwen/Qwen2.5-7B-Instruct
```

To pull them ahead of time rather than on first use:

```bash
hf download facebook/contriever-msmarco
hf download Qwen/Qwen2.5-0.5B-Instruct
```

### Data: one command

`data/` is deliberately not in git, so a fresh clone has no datasets. Fetch
them, which also verifies every split against Table 5 of the paper:

```bash
python -m semproject.setup_data
```

About 519 MB, most of it XSum. Pass a subset to skip the rest:
`python -m semproject.setup_data bbh fp`.

Seven of the eight tasks need nothing further. GPQA is gated: accept the licence
at `Idavidrein/gpqa`, run `hf auth login`, then re-run the script.

### Full install

```bash
git clone https://github.com/LuisPostigo/MAPLE_ICL.git
cd MAPLE_ICL && git checkout ml-semProject
python3 -m venv .venv
.venv/bin/pip install torch transformers datasets networkx numpy scipy \
    scikit-learn pandas rouge tqdm peft accelerate
.venv/bin/python -m semproject.setup_data
.venv/bin/python -m semproject.test_smoke
```

The smoke tests need no model weights and no network, so they are the fastest
check that the install is sound. Expect roughly 3 GB total once the models have
been pulled.

## The model boundary

Nothing outside `model/` imports a backend. Everything goes through
`get_model(name)` and touches only `generate(prompt, temperature, max_tokens)`,
which returns a `Response` carrying `text` and, where the backend can supply
them, per-token `logprobs`. Adding a backend is one subclass plus one registry
line.

| Backend | Logprobs | Notes |
|---|---|---|
| `mock` | synthetic | deterministic, offline, no weights — for exercising the pipeline |
| `hf` | real | local transformers; defaults to Qwen2.5-0.5B-Instruct |
| `gemini` | none | API; confidence falls back to a JSON self-report |

Only `hf` gives true token logprobs, which is what the confidence signal wants.

## The gate

Box 4 of the design. Signals produce `f_1..f_n`, a combiner `G` maps them to
`r(x, ŷ) ∈ [0,1]`, and a threshold decides:

    r ≥ τ  →  keep in the reliable demonstration pool
    r < τ  →  filter out, or route to re-labeling

| Signal | v0 implementation | Status |
|---|---|---|
| `confidence.py` | mean token logprob → `exp`; JSON self-report when unavailable | working |
| `consistency.py` | re-query k times at temperature > 0, fraction matching | working |
| `influence.py` | MAPLE's graph influence on the labeled set, normalised | **placeholder** |

`weighted_mean` in `gate.py` is a placeholder for `G`. Replace it with a learned
combiner by passing `combine=`; nothing else changes.

The influence signal is a placeholder in the sense that matters: the intended
version measures how much a demonstration changes model behaviour, in the spirit
of influence functions where the effect scales with learning rate. That needs
gradient access and cannot run against an API. v0 substitutes the graph
influence MAPLE has already computed, which costs no model calls.

## Running

```bash
# Task 1 — reproduce MAPLE, no gate
.venv/bin/python -m semproject.run -t date -m mock --no-gate

# Task 2 — add the gate, with a real local model
.venv/bin/python -m semproject.run -t fp -m hf -nl 8 -nu 16 --limit-test 20
```

`--signals`, `--weights` and `--threshold` configure the gate. `--out FILE`
appends JSON lines.

## What the model is actually asked to label

The options are never generated. For the multiple-choice tasks they arrive with
the question as part of the input; for the classification tasks the taxonomy is
a fixed list printed into every prompt. The model only ever emits the label.

That said, the three regimes are not equally useful for reliability work.

| Task | Model emits | Label space | Chance of a correct guess |
|---|---|---|---|
| date | `(A)` | 6 options, shipped with the question | 17% |
| salient | `(A)` | 6 options, shipped with the question | 17% |
| tracking | `(F)` | 7 options, shipped with the question | 14% |
| gpqa | `(A)` | 4 options, shipped with the question | 25% |
| fp | `neutral` | 3 fixed classes | 33% |
| goemo | `neutral` | 28 fixed classes | 3.6% |
| banking77 | `card_arrival` | 77 fixed classes | 1.3% |
| xsum | a one-sentence summary | open generation | ~0% |

**Letter pick** — the options are part of `x`, so the label is one token:

```
Give only the choice the correct answer by selecting one of the options (e.g., '(A)', '(B)').
Question: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
Options:
(A) 05/01/2021
(B) 02/23/2021
...
```

**Exact string from a fixed taxonomy** — the 77 class names are listed in every
prompt and the model must reproduce one verbatim:

```
You can only make prediction from the following categories: activate_my_card,
age_limit, ... wrong_exchange_rate_for_cash_withdrawal.
service query: I am still waiting on my card?
intent category:
```

**Open generation** — no label space at all, scored by ROUGE-L:

```
Give only the summary, and no extra commentary.
Article: Clean-up operations are continuing across the Scottish Borders ...
```

### Why this matters for the gate

The rightmost column above is the problem. On a 6-way task roughly one in six
wrong guesses lands on the correct answer anyway, so a pseudo-label can be
"correct" by accident. Those lucky guesses sit in the positive class of the
detection AUC, and no reliability signal can distinguish them from labels the
model actually knew. That is noise in the target variable, not in the features,
and it caps how well any signal can score.

Banking77 drops chance agreement to 1.3% and XSum effectively to zero, which
makes pseudo-label correctness mean what we want it to mean. They also make the
signals richer: confidence over a 77-token class name carries more than the
logprob of a single letter, and consistency over 77 outcomes has a far lower
floor than consistency over 6.

### The fix, in two parts

**Banking77 is now the default task.** It keeps full comparability with the
paper, which uses it as a headline result, while dropping chance agreement to
1.3 percent. The model must reproduce one of 77 class names exactly, so being
correct means it knew the answer. `date` was the wrong default both for the
chance rate and because its training pool is 98.6 percent gold (A).

**`--open-labels` hides the answer options**, so the label has to be generated
rather than picked. Scoring matches the generated text against the gold option
text, accepting the answer anywhere in the reply.

```bash
python -m semproject.run -t date --open-labels
```

Only worth using where the answer space is actually large:

| Task | Distinct gold answers in the pool | Useful with `--open-labels` |
|---|---|---|
| date | 322 | yes, becomes genuinely generative |
| tracking | 47 | partly |
| salient | 6 | no, stays a closed set either way |

Early evidence that the chance rate was the binding problem: on the same tiny
configuration, gate detection AUC is 0.50 on Financial PhraseBank (33 percent
chance) and 0.78 on Banking77 (1.3 percent chance). Both runs are far too small
to conclude anything, but the direction is the predicted one.

## How we know the gate works

Pseudo-labeled samples come from a training split that has ground truth; MAPLE
never looks at it. We withhold it during labeling and reveal it only for
scoring, so `pseudo_label_accuracy` and per-signal detection AUC cost no extra
model calls.

`signal_auc` is the number that matters: the probability a signal ranks a
correct pseudo-label above a wrong one. **0.5 is chance.** A signal at 0.5
cannot filter anything, and should be dropped before any method is built on it.

## One upstream scorer bug you need to know about

`process_bbh.normalize_answer` strips the article "a", so `"(A)"` becomes the
empty string. On gold-(A) items this **inverts** scoring, because `"" in
anything` is true:

| gold | predicted | upstream | correct |
|---|---|---|---|
| (A) | (A) | 0 | 1 |
| (A) | (B) | 1 | 0 |
| (B) | (B) | 1 | 1 |

It rewards wrong answers and punishes right ones. Date is hit hardest: its
training pool is 98.6% gold (A), so pseudo-label accuracy there is meaningless
under the upstream scorer — it reported 1.0 before we caught this.

`data.score` corrects it by default. Pass `faithful=True` to reproduce the
paper's published numbers.
