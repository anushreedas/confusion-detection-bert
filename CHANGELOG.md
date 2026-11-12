# Changelog

## 0.1.0
- Re-engineered the 2021 capstone notebooks into an installable, tested package (`confusion_bert`).
- Fixed: duplicate-sentence merge bug, empty-sentence crash, non-deterministic file order, unseeded subsampling,
  removed-API usages. Full list in `docs/METHODOLOGY.md`.
- Added: typed YAML configs with CLI overrides, group-wise (speaker-independent) split option, best-epoch
  checkpointing, macro-F1 / per-class metrics, exact McNemar + paired bootstrap + Holm correction beside the original
  sign test, inference API, optional FastAPI service, Dockerfile, CI.
