# Methodology and differences from the 2021 capstone code

This repository is a re-engineering of a 2021 course capstone (`notebooks/legacy/`, `docs/capstone_report_2021.pdf`).
The scientific design is preserved; where the original notebooks contained bugs, undocumented behaviour or claims
that the code did not support, the repository fixes or documents them. This page lists every deliberate difference.

## Pipeline

1. **Align labels to tokens.** Decode change points `(decisecond, state)`, convert to seconds (`ceil(t/10)`), give each
   token the state of the latest change point at or before its timestamp (`(0, 0)` always present).
2. **Segment into sentences** on `.` / `?` tokens; sentence label = **max** token label.
3. **De-duplicate** sentences globally (first occurrence wins) so identical utterances cannot straddle splits.
4. **Split** into train / val / test (`stratified` sentence-level, as in the original, or `group` = held-out participants).
5. **Balance** train and val by downsampling every class to the smallest class. Test is left at its natural distribution.
6. **Fine-tune** `encoder → [CLS] → Linear(H,50) → ReLU → Linear(50,4)`; AdamW, lr 5e-5, 200 warm-up steps, linear decay,
   clip 1.0, 4 epochs, batch 16, cross-entropy.
7. **Compare** models with significance tests (below).

## Differences from the original notebooks

| # | Original behaviour | Issue | This repository |
|---|---|---|---|
| 1 | When a sentence was a duplicate, the terminator was appended instead of closing the sentence | The duplicate's tokens and `.` merged into the *next* sentence, corrupting text and labels | Segmentation and de-duplication are separate steps; regression test in `tests/test_sentences.py` |
| 2 | Empty sentence (two terminators in a row) → `max([])` | Crash | Skipped |
| 3 | README: "stratified 74/16/10"; report: "70:20:10"; code: `train_test_split` per file with no stratification (≈75.6/14.4/10) | Three inconsistent descriptions; no stratification; split per participant file | One documented protocol: global split, stratified by default, `val_fraction_of_remaining=0.16` ⇒ ≈75.6/14.4/10. `group` option holds out whole participants |
| 4 | Files processed in `os.walk` order | De-duplication winner depended on filesystem order | Sorted, deterministic |
| 5 | No seed in the 20 test subsamples | Not reproducible | Seeded; the same subsamples are shared by every model |
| 6 | `BertModel` loaded twice per model, hard-coded 768-d head | Waste; breaks for other encoders | Single load; head size from `config.hidden_size` |
| 7 | `transformers.AdamW`, `datasets.load_metric`, `scipy.stats.binom_test` | Removed in current library versions | `torch.optim.AdamW`, own metrics, `scipy.stats.binomtest` |
| 8 | Whole dataset padded to a single length (up to 512) | Slow | Dynamic per-batch padding (numerically equivalent under attention masking) |
| 9 | Last-epoch weights saved | Ignores validation signal | `select: best_val_macro_f1` by default (`select: last` restores the original) |
| 10 | Inference without `torch.no_grad()`, no device handling | Memory-hungry, CPU-only | `ConfusionPredictor` |
| 11 | Pooler instantiated but unused | Dead parameters | Not instantiated (outputs unchanged) |
| 12 | `12freeze6` configuration not present in the notebook code | Not reproducible from the code | Implemented from Fig. 4 of the report as *freeze embeddings + first 6 encoder layers*. **Inferred, not copied** |

## What "14 and 18 layers" means

`bert-base-uncased` has 12 pretrained layers. The original code set `config.num_hidden_layers` to 2/6/14/18. For 2 and 6
the encoder is **truncated** (only the first N pretrained layers are loaded). For 14 and 18 the extra 2 / 6 layers have
**no pretrained weights and are randomly initialised**. The repository logs this explicitly. "More layers" is therefore a
statement about adding fresh capacity on top of the pretrained stack, not about a deeper pretrained model.

## Statistical comparison

The original protocol: draw 20 stratified random subsamples (30%) of the test set, score each model on each, run a
one-sided **sign test** on the paired scores (Dror et al., 2018). Reproduced in `evaluation/significance.py`.

Caveats a careful reader should know:

* **Subsamples overlap** (each is 30% of the same test set), so the 20 scores are not independent. The sign test
  assumes i.i.d. pairs; its p-values are optimistic. A model that wins all 20 subsamples always gets `0.5**20 ≈ 9.5e-7`
  — the "0.0000009" values in the report — regardless of effect size.
* **Many pairwise tests, no correction** in the original. Here Holm-adjusted p-values are reported alongside.
* **Accuracy on an imbalanced test set** is dominated by class 0. Macro-F1 and per-class metrics are reported by this
  repository; the original reported accuracy only.

Complementary tests on the *full* test set are therefore provided: **exact McNemar** on per-example correctness and a
**paired bootstrap** CI for the metric difference.

## Known limitations (inherited from the study design)

* **Sentence-level random splits** can place utterances from the same speaker/conversation in train and test; use
  `split_strategy: group` for a speaker-independent estimate. Reported 2021 numbers use the random protocol.
* **Model selection on the test set**: the 2021 conclusion ("BERT-base-6 is best") compares test-set scores of 12
  configurations. A clean protocol would select on validation and report test once.
* **Max-aggregation of labels** makes long sentences more likely to receive a higher label.
* **Small data** (hundreds of balanced training examples; 20 participants). Differences of 1–2 accuracy points between
  configurations are within what seed variance alone could produce; a single training run per configuration was used.
