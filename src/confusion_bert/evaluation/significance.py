"""Statistical comparison of classifiers evaluated on the same test set.

Three complementary tests are provided:

* **Sign test on repeated stratified subsamples** — the protocol of the original
  capstone (Dror et al., 2018). Caveat: subsamples drawn from one test set overlap,
  so the per-subsample scores are *not* independent and the p-values are
  optimistic (e.g. 20/20 wins always gives 0.5**20 ~= 9.5e-7).
* **Exact McNemar test** on per-example correctness over the full test set.
* **Paired bootstrap** confidence interval for the metric difference.

Pairwise p-values can be Holm-adjusted for multiple comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

from confusion_bert.evaluation.metrics import accuracy, macro_f1

Metric = Callable[[np.ndarray, np.ndarray], float]
METRICS: dict[str, Metric] = {"accuracy": accuracy, "macro_f1": macro_f1}


def make_stratified_subsamples(
    labels: ArrayLike, n_subsamples: int = 20, fraction: float = 0.3, seed: int = 0
) -> list[np.ndarray]:
    """Index sets of stratified random subsamples (shared across all compared models)."""
    labels_arr = np.asarray(labels)
    idx = np.arange(len(labels_arr))
    out = []
    for i in range(n_subsamples):
        chosen, _ = train_test_split(
            idx, train_size=fraction, stratify=labels_arr, random_state=seed + i
        )
        out.append(np.sort(chosen))
    return out


def subsample_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_sets: Sequence[np.ndarray],
    metric: str = "accuracy",
) -> np.ndarray:
    fn = METRICS[metric]
    return np.array([fn(y_true[ix], y_pred[ix]) for ix in index_sets])


@dataclass(frozen=True)
class SignTestResult:
    p_value: float
    wins: int
    losses: int
    ties: int


def sign_test(scores_a: ArrayLike, scores_b: ArrayLike) -> SignTestResult:
    """One-sided exact sign test, H1: A tends to score higher than B. Ties are dropped."""
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    wins, losses, ties = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    n = wins + losses
    p = 1.0 if n == 0 else float(binomtest(wins, n, 0.5, alternative="greater").pvalue)
    return SignTestResult(p, wins, losses, ties)


@dataclass(frozen=True)
class McNemarResult:
    p_value: float
    a_only_correct: int
    b_only_correct: int


def mcnemar_exact(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, alternative: str = "greater"
) -> McNemarResult:
    """Exact McNemar test on discordant pairs. ``greater``: A more often right than B."""
    ca, cb = pred_a == y_true, pred_b == y_true
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(b, n, 0.5, alternative=alternative).pvalue)
    return McNemarResult(p, b, c)


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float  # one-sided, H1: A > B


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str = "accuracy",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap over test examples for ``metric(A) - metric(B)``."""
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        ix = rng.integers(0, n, n)
        diffs[i] = fn(y_true[ix], pred_a[ix]) - fn(y_true[ix], pred_b[ix])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = (int((diffs <= 0).sum()) + 1) / (n_resamples + 1)
    return BootstrapResult(
        float(fn(y_true, pred_a) - fn(y_true, pred_b)), float(lo), float(hi), float(p)
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def pairwise_compare(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    n_subsamples: int = 20,
    subsample_fraction: float = 0.3,
    bootstrap_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Compare every ordered pair (A, B) of models; one row per pair.

    ``*_p`` columns test H1 "A is better than B". ``sign_p_holm`` adjusts the
    sign-test p-values across all pairs in the table.
    """
    y_true = np.asarray(y_true)
    index_sets = make_stratified_subsamples(y_true, n_subsamples, subsample_fraction, seed)
    scores = {m: subsample_scores(y_true, p, index_sets, metric) for m, p in predictions.items()}
    rows = []
    for a in predictions:
        for b in predictions:
            if a == b:
                continue
            st = sign_test(scores[a], scores[b])
            mc = mcnemar_exact(y_true, predictions[a], predictions[b])
            bs = paired_bootstrap(
                y_true, predictions[a], predictions[b], metric, bootstrap_resamples, alpha, seed
            )
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    f"{metric}_a": float(METRICS[metric](y_true, predictions[a])),
                    f"{metric}_b": float(METRICS[metric](y_true, predictions[b])),
                    "sign_wins": st.wins,
                    "sign_losses": st.losses,
                    "sign_ties": st.ties,
                    "sign_p": st.p_value,
                    "mcnemar_p": mc.p_value,
                    "boot_diff": bs.mean_diff,
                    "boot_ci_low": bs.ci_low,
                    "boot_ci_high": bs.ci_high,
                    "boot_p": bs.p_value,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sign_p_holm"] = holm_adjust(df["sign_p"].tolist())
        df["mcnemar_p_holm"] = holm_adjust(df["mcnemar_p"].tolist())
    return df
