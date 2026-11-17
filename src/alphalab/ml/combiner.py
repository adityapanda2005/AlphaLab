"""Combining many weak alphas into one forecast.

The individual alphas that survived selection each carry a little information
and a lot of noise.  The combiner's job is to weigh them against each other --
and the hard part is doing so without quietly fitting the noise.

Design decisions, and why
-------------------------
*Cross-sectional standardisation of the features.*  Each alpha is z-scored
across instruments within each date before the model sees it.  Alphas live on
wildly different scales (a rank is in [0,1], a volatility ratio is unbounded),
and without this the regularisation penalty would fall almost entirely on
whichever alpha happened to have the largest units.  Standardising *within a
date* rather than globally also means no statistic is computed across time, so
the transform cannot leak information from the future.

*A rank target.*  The label is the cross-sectional rank of the forward return,
centred on zero, rather than the raw return.  Raw returns are heavy-tailed:
fit a squared-error model to them and a handful of earnings-day moves dominate
the coefficients.  Since the portfolio is built from the *ordering* of the
forecast anyway, predicting the ordering directly is both more robust and
better matched to how the output is used.

*Ridge first, gradient boosting second.*  Ridge is the honest baseline: linear,
convex, one hyper-parameter, and almost impossible to accidentally overfit with
a strong penalty.  LightGBM can capture interactions between alphas -- reversal
behaving differently in high-volatility names, say -- but has far more capacity
to memorise.  Reporting both is the point: if the expensive model does not beat
the cheap one out of sample, the honest conclusion is that there is no
interaction worth modelling, and the README reports whichever result occurs.

*Every fit happens inside a fold.*  Standardisation statistics, model
coefficients and the feature list are all derived from training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .splits import Fold, PurgedWalkForward, assert_no_leakage


@dataclass
class CombinerConfig:
    """Settings for the signal-combination step."""

    model: str = "ridge"            # "ridge", "lightgbm" or "equal_weight"
    horizon: int = 1
    n_splits: int = 6
    embargo: int = 5
    min_train: int = 504
    expanding: bool = True
    ridge_alpha: float = 50.0       # strong by default: inputs are collinear
    target: str = "rank"            # "rank" or "return"
    winsorize_features: float = 4.0
    random_state: int = 7

    # LightGBM settings, kept small on purpose.  The signal-to-noise ratio
    # here is a few percent, so a deep forest will memorise the sample.
    lgbm_n_estimators: int = 300
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 15
    lgbm_min_child_samples: int = 200
    lgbm_subsample: float = 0.7
    lgbm_colsample: float = 0.7
    lgbm_reg_lambda: float = 10.0


@dataclass
class CombinerResult:
    """Out-of-sample forecasts plus a record of how they were produced."""

    predictions: pd.DataFrame                     # dates x instruments
    folds: List[Fold] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    coefficients: Optional[pd.DataFrame] = None   # one row per fold
    importances: Optional[pd.DataFrame] = None
    fold_scores: Optional[pd.DataFrame] = None
    config: Optional[CombinerConfig] = None

    @property
    def mean_coefficients(self) -> Optional[pd.Series]:
        if self.coefficients is None or self.coefficients.empty:
            return None
        return self.coefficients.mean().sort_values(
            key=lambda s: s.abs(), ascending=False
        )

    def describe(self) -> str:
        covered = int(self.predictions.notna().any(axis=1).sum())
        return (
            f"{self.config.model if self.config else '?'} combiner: "
            f"{len(self.folds)} folds, {len(self.feature_names)} features, "
            f"{covered} out-of-sample dates"
        )


# ----------------------------------------------------------------------
# Feature and label preparation
# ----------------------------------------------------------------------
def build_design_matrix(
    signals: Dict[str, pd.DataFrame],
    names: Sequence[str],
    *,
    mask: Optional[pd.DataFrame] = None,
    winsorize: float = 4.0,
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    """Stack selected alphas into a 3-D array of (date, instrument, feature).

    Each alpha is standardised across instruments within each date.  Returning
    a dense array rather than a long DataFrame keeps the per-fold slicing to
    simple numpy views, which matters because this is done once per fold per
    model.
    """
    names = list(names)
    if not names:
        raise ValueError("no features selected")

    reference = signals[names[0]]
    dates, instruments = reference.index, reference.columns

    stacked = np.full((len(dates), len(instruments), len(names)), np.nan)
    for position, name in enumerate(names):
        frame = signals[name].reindex(index=dates, columns=instruments)
        if mask is not None:
            frame = frame.where(mask.reindex(index=dates, columns=instruments))
        stacked[:, :, position] = _cross_sectional_zscore(
            frame.to_numpy(dtype=float), winsorize
        )

    return stacked, dates, instruments


def _cross_sectional_zscore(values: np.ndarray, winsorize: float) -> np.ndarray:
    """Standardise each row, then clip extremes.

    Clipping after standardising rather than before means the threshold is in
    units of that date's own dispersion, so it adapts to calm and turbulent
    markets instead of imposing one fixed cut-off.
    """
    import warnings

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=1, keepdims=True)
        std = np.nanstd(values, axis=1, keepdims=True)
    std = np.where(std > 0, std, np.nan)
    out = (values - mean) / std
    if winsorize and winsorize > 0:
        out = np.clip(out, -winsorize, winsorize)
    return out


def build_target(
    forward_returns: pd.DataFrame,
    *,
    kind: str = "rank",
    mask: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Build the per-date prediction target.

    ``"rank"`` gives the centred cross-sectional percentile rank, in roughly
    [-0.5, 0.5]; ``"return"`` gives the cross-sectionally demeaned raw return.
    """
    frame = forward_returns
    if mask is not None:
        frame = frame.where(mask.reindex_like(frame))

    if kind == "rank":
        ranked = frame.rank(axis=1, pct=True, na_option="keep")
        return (ranked - 0.5).to_numpy(dtype=float)
    if kind == "return":
        return frame.sub(frame.mean(axis=1), axis=0).to_numpy(dtype=float)
    raise ValueError("target must be 'rank' or 'return'")


# ----------------------------------------------------------------------
# The combiner
# ----------------------------------------------------------------------
def combine_signals(
    signals: Dict[str, pd.DataFrame],
    names: Sequence[str],
    panel,
    *,
    config: Optional[CombinerConfig] = None,
    mask: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> CombinerResult:
    """Fit the combiner walk-forward and return out-of-sample forecasts.

    Every value in ``result.predictions`` was produced by a model that saw only
    data from strictly before that date, separated by the purge and embargo
    gap.  That is what makes the backtest built on top of it meaningful.
    """
    config = config or CombinerConfig()
    names = list(names)

    features, dates, instruments = build_design_matrix(
        signals, names, mask=mask, winsorize=config.winsorize_features
    )
    forward = panel.forward_returns(config.horizon)
    target = build_target(
        forward.reindex(index=dates, columns=instruments),
        kind=config.target,
        mask=mask,
    )

    splitter = PurgedWalkForward(
        n_splits=config.n_splits,
        horizon=config.horizon,
        embargo=config.embargo,
        min_train=config.min_train,
        expanding=config.expanding,
    )
    folds = splitter.split(dates)
    assert_no_leakage(folds, config.horizon)

    predictions = np.full((len(dates), len(instruments)), np.nan)
    coefficient_rows: List[dict] = []
    importance_rows: List[dict] = []
    score_rows: List[dict] = []

    for fold in folds:
        x_train, y_train = _flatten(
            features[fold.train_start : fold.train_end],
            target[fold.train_start : fold.train_end],
        )
        if len(y_train) < 100:
            continue

        model = _fit_model(x_train, y_train, config)

        test_block = features[fold.test_start : fold.test_end]
        predicted = _predict_block(model, test_block)
        predictions[fold.test_start : fold.test_end] = predicted

        if verbose:
            print(f"  {fold.describe(dates)}  n_train={len(y_train):,}")

        score_rows.append(
            {
                "fold": fold.index,
                "n_train": int(len(y_train)),
                "train_start": dates[fold.train_start],
                "test_start": dates[fold.test_start],
                "test_end": dates[fold.test_end - 1],
                "train_ic": _in_sample_ic(model, x_train, y_train),
            }
        )

        if hasattr(model, "coef_"):
            coefficient_rows.append(
                {"fold": fold.index, **dict(zip(names, model.coef_))}
            )
        if hasattr(model, "feature_importances_"):
            importance_rows.append(
                {
                    "fold": fold.index,
                    **dict(zip(names, model.feature_importances_)),
                }
            )

    result = CombinerResult(
        predictions=pd.DataFrame(
            predictions, index=dates, columns=instruments
        ),
        folds=folds,
        feature_names=names,
        config=config,
        fold_scores=pd.DataFrame(score_rows) if score_rows else None,
    )
    if coefficient_rows:
        result.coefficients = pd.DataFrame(coefficient_rows).set_index("fold")
    if importance_rows:
        result.importances = pd.DataFrame(importance_rows).set_index("fold")
    return result


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def _fit_model(x: np.ndarray, y: np.ndarray, config: CombinerConfig):
    if config.model == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
        model.fit(x, y)
        return model

    if config.model == "equal_weight":
        return _EqualWeight(n_features=x.shape[1])

    if config.model == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'the lightgbm model needs: pip install -e ".[ml]"'
            ) from exc

        model = LGBMRegressor(
            n_estimators=config.lgbm_n_estimators,
            learning_rate=config.lgbm_learning_rate,
            num_leaves=config.lgbm_num_leaves,
            min_child_samples=config.lgbm_min_child_samples,
            subsample=config.lgbm_subsample,
            subsample_freq=1,
            colsample_bytree=config.lgbm_colsample,
            reg_lambda=config.lgbm_reg_lambda,
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x, y)
        return model

    raise ValueError(f"unknown model {config.model!r}")


class _EqualWeight:
    """Average the standardised alphas, with no fitting at all.

    The most honest baseline in the project.  If a fitted model cannot beat
    the simple average of its own inputs out of sample, the fitting added
    nothing, and saying so is more useful than quietly reporting the fitted
    number.
    """

    def __init__(self, n_features: int) -> None:
        self.coef_ = np.full(n_features, 1.0 / n_features)
        self.intercept_ = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.nanmean(x, axis=1)


def _predict_block(model, block: np.ndarray) -> np.ndarray:
    """Predict for one test fold, preserving the (date, instrument) shape."""
    n_dates, n_instruments, n_features = block.shape
    flat = block.reshape(-1, n_features)

    usable = ~np.isnan(flat).any(axis=1)
    out = np.full(flat.shape[0], np.nan)
    if usable.any():
        out[usable] = model.predict(flat[usable])
    return out.reshape(n_dates, n_instruments)


def _flatten(
    features: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows with any missing feature or a missing label.

    Dropping rather than imputing is the right call here: a missing alpha
    usually means the instrument has too little history for that lookback, and
    imputing a zero would tell the model "this signal is exactly average",
    which is a statement the data does not support.
    """
    n_features = features.shape[-1]
    x = features.reshape(-1, n_features)
    y = target.reshape(-1)

    keep = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
    return x[keep], y[keep]


def _in_sample_ic(model, x: np.ndarray, y: np.ndarray) -> float:
    """Correlation of fitted values with the training target.

    Reported next to each fold purely as a diagnostic: a large gap between
    this and the out-of-sample IC is the signature of overfitting.
    """
    if len(y) < 10:
        return float("nan")
    fitted = model.predict(x)
    if np.std(fitted) == 0:
        return float("nan")
    return float(np.corrcoef(fitted, y)[0, 1])
