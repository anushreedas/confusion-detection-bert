# Data

**No transcripts, audio or video are included in this repository, and none should be committed.**

## What the study data is

The confusion data was collected in an earlier study (Kaushik et al., 2021; Mince et al., 2021 — see the
capstone report in `docs/`). Participants completed confusion-eliciting tasks over Zoom and then annotated
their own recordings with one of four levels: `0` not confused, `1` slightly, `2` very, `3` extremely confused.
Only the **transcribed speech** is used here.

## Expected layout

```
data/raw/
  t1/903.csv     # <session dir ending in the session number>/<participant id>.csv
  t2/903.csv
  t3/903.csv
  ...
```

Each CSV has a header row, then one word token per row. The first two columns are used:

| column 1 | column 2 | column 3 (ignored) |
|---|---|---|
| word (`.` and `?` act as sentence terminators) | start time in **seconds** | end time |

These column semantics are inferred from the original notebooks; adjust `read_transcript` if your files differ.

## Labels

`data/labels/conf_labels.json` maps `"<participant>t<session>"` to an encoded annotation string, extracted
verbatim from the original `Data Preprocessing.ipynb`. Decoding it needs the course-provided `base85`
module (`time16_decode`), which is **not** distributed here. Install it, or write your own decoder and set
`data.label_decoder: "your_module:your_function"` (returns `[(decisecond, state), ...]`).

> **Licensing / consent.** The label file derives from human-participant annotations. It is kept here because
> it was already part of the original public capstone repository, but you should confirm you are allowed to
> redistribute it before publishing. The MIT licence of this repository covers source code only.

## Producing the splits

```bash
confusion-bert prepare --config configs/base.yaml
```

writes `data/processed/{sentences,train,val,test,train_balanced,val_balanced}.csv` (git-ignored).
