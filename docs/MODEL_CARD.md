# Model card — confusion-bert

## Summary
Sentence-level classifier that predicts one of four self-reported confusion levels from a single transcribed
utterance. Encoder: `bert-base-uncased` or `TODBERT/TOD-BERT-JNT-V1` (optionally truncated/extended/partially
frozen) with a 768→50→4 feed-forward head. **No trained weights are distributed with this repository.**

## Intended use
Research and teaching: studying linguistic cues of confusion, comparing fine-tuning strategies, demonstrating
significance testing for small-data NLP comparisons.

## Not intended for
* Decisions about individuals (hiring, grading, proctoring, performance management, clinical or legal use).
* Any setting where people have not given informed consent to affect inference.
* Languages other than English, or text that was not produced as spoken conversation.

## Training data
Self-annotated Zoom conversations from an earlier study (≈20 participants, three tasks each), transcribed to
timestamped words. Labels are *the participant's own retrospective judgement* of how confused they were, which is
subjective and noisy. Sentence label = max over the sentence's time span. See `data/README.md`.

## Evaluation
Original 2021 results (accuracy on repeated 30% test subsamples, single run per configuration, test set also used
for model selection) are listed in the README and `results/reported_2021.csv`. They have **not** been re-run in this
repository because the data and the label decoder are not distributed. Treat them as a claim from the original study,
not as a verified benchmark. Use `confusion-bert compare` for the stricter tests described in `docs/METHODOLOGY.md`.

## Risks and limitations
* Self-reported labels reflect individual and cultural differences in expressing or admitting confusion.
* Disfluencies ("um", "uh") are informative cues but are speaker-dependent; the model may be biased toward speakers
  or accents that produce more of them.
* Few participants ⇒ high variance and probable speaker leakage under the random-split protocol.
* The fully frozen encoder collapses to near chance; some fine-tuning is required.
