# Pre-trained HF Fine-Tune — Strategic Review

The question on the table: instead of training a small transformer from
scratch on the ~3,000 Infineon sequences, should we **fine-tune a pre-trained
model from Hugging Face** (e.g. a small Llama / Qwen / Mistral, or a smaller
sequence model like GPT-2-small)?

## Short answer

> **No, not as the primary submission.** Pre-train-then-fine-tune is the
> wrong shape for this benchmark. Add a **small from-scratch decoder-only
> transformer** as the primary improved model. *Optionally* keep a single HF
> fine-tune as a "stretch / contrast" experiment — it makes a strong slide,
> but it should not be the load-bearing submission.

## Why a from-scratch small transformer wins on this benchmark

### 1. The vocabulary is unrelated to natural language
The 120-token Infineon vocabulary (`RECEIVE WAFER LOT`, `RCA CLEAN 1`,
`EXPOSE LITHO LEVEL 3`, …) maps to **arbitrary multi-subword BPE chunks** in
any natural-language tokenizer. A pre-trained model has zero transferable
prior on these token relationships — its embeddings for them are essentially
random, the same as a from-scratch model, but you now also pay the cost of
huge embedding matrices for the other 30k–250k tokens you'll never use.

A from-scratch model defines `len(vocab) ≈ 130` and gets a 130 × d_model
embedding table instead of (say) 50,000 × d_model. With d_model=128, that is
~16k parameters vs ~6.4M, a **400× reduction** in just the embedding layer.

### 2. The track explicitly rewards "open, transparent stack"
From `Track_industrial_en.md` §4 (Model Specification, Constraints):

> Focus on reproducible training and evaluation setups on cluster
> infrastructure and on an open, transparent stack rather than black-box API
> solutions.

A 4–6 layer decoder transformer trained from scratch on a 130-token vocab is
the most transparent thing you can build. Showing scaling curves
(small/medium, 3k/30k sequences) directly serves the "scaling effects"
stretch goal. A fine-tuned 7B is a black box from the judges' perspective.

### 3. The benchmark is designed to detect *memorization*
Task 4 (post-submission OOD generalization on a hidden 4th product family)
specifically penalizes "memorize then look up." A model that has seen
billions of English tokens has more capacity to memorize and re-emit
fragments of the dev set than a 5M-param model trained for one hour. The
judge story for OOD performance is much stronger when the architecture is
provably tiny.

### 4. Speed and reproducibility on 36 hours
- A small decoder-only transformer with d_model=128, n_layers=4, max_len=180
  trains to convergence on the 3k base corpus in **minutes** on a single
  A100 — leaving compute for the scaling study (`small+data`, `medium+data`).
- A 1B LoRA fine-tune on the same corpus is hours per run, with HF tokenizer
  / template debugging time on top, on a Leonardo environment you haven't
  validated yet.

### 5. Hybrid system already covers the symbolic axis
Task 3 is best carried by the **explicit rule validator** (we already wired
the official `validate_sequence` — F1 ≈ 1.0 on dev). A large language model
cannot beat that, and trying to is a trap: even a 99% LLM classifier on Task
3 is worse than 100% rules, and it adds variance to the demo.

## When the HF fine-tune *would* make sense

- If the track allowed and rewarded **multi-modal context** — e.g. if Tasks
  1/2 input included free-text `DESCRIPTION` strings from
  `*_Longdescr.csv` and you needed text understanding. It doesn't.
- If the eval expected **natural-language explanations** of anomaly causes.
  It doesn't.
- If you had 100k+ sequences and limited compute. You have the opposite:
  small data, generous compute.

## Recommended use of HF — as a labeled stretch experiment

If you have spare time after the from-scratch transformer is trained and the
demo is wired, run **one** HF experiment and report it side-by-side. This
buys the "we honestly tried both directions" slide:

1. **Model:** `distilgpt2` (82M params, smallest credible GPT-style model on
   the Hub, very fast to fine-tune). Avoid Llama / Qwen / Mistral — too
   large, license drag, and they shift the conversation away from "open
   transparent stack."
2. **Tokenization:** wrap each step token in a special marker like
   `<step:RECEIVE_WAFER_LOT>` and add all ~130 markers as additional
   tokens via `tokenizer.add_tokens(...)`. Resize embeddings. This stops
   BPE from fracturing step names and gives the comparison a fair shake.
3. **Training:** standard causal LM loss on the same train split as the
   from-scratch model. 1–3 epochs.
4. **Eval:** the same `evaluate_all(model, eval_dir)` — wrap the HF model in
   a class that implements `predict_topk` + `complete`. Zero new evaluator
   code needed.
5. **Report:** put `from-scratch-small` and `distilgpt2-finetune` in the
   same scaling table. The story the judges want to hear is "even when we
   start from a pre-trained LM, the from-scratch transformer on this
   tightly-scoped vocabulary holds its own / wins on OOD."

## What NOT to do

- **Don't** use the OpenAI / Anthropic API at inference time. The brief
  explicitly contrasts this with "a pure API or wrapper approach" and rules
  it out as the focus.
- **Don't** fine-tune anything > 1B params. Leonardo time is finite and the
  ROI is poor on a 130-token vocab.
- **Don't** let the HF experiment hold up the from-scratch + n-gram +
  validator + demo path. It is *contrast material*, not the spine of the
  submission.

## Decision

> **Spine:** n-gram baseline (done ✅) → small from-scratch decoder-only
> transformer (next PR) → rule validator for Task 3 (done ✅) → Streamlit
> side-by-side demo (after transformer lands).
>
> **Stretch contrast:** one `distilgpt2` fine-tune evaluated via the same
> harness, reported in the scaling table. Only after the spine is green.
