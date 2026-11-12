# Legacy notebooks (2021 capstone)

Preserved **unmodified** for provenance. They are not part of the tested package and some cells will not run on
current library versions (`datasets.load_metric`, `transformers.AdamW`, `scipy.stats.binom_test` were removed).
The maintained implementation lives in `src/confusion_bert/`; see `docs/METHODOLOGY.md` for every difference.

| Notebook | Superseded by |
|---|---|
| Data Preprocessing | `confusion_bert.data.*`, `confusion-bert prepare` |
| Fine-tune Bert models | `confusion_bert.training`, `confusion-bert train` / `sweep` |
| Sampling | `evaluation.significance.make_stratified_subsamples` |
| Predict | `confusion_bert.inference`, `confusion-bert evaluate` |
| Accuracy Scores | `evaluation.metrics`, `evaluation.significance.subsample_scores` |
| Signed Test Final | `evaluation.significance`, `confusion-bert compare` |
