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
