# AlphaLab

**Alpha Research & Backtesting Engine**

A self-contained pipeline that takes a library of quantitative alpha formulas, evaluates them against market data, selects the ones with genuine predictive power, combines them with machine learning, backtests the result — and corrects for the multiple-testing bias that makes most backtests unreliable.

```
alphalab/
├── expr/          Formula DSL: tokenizer → parser → AST → cached evaluator
├── operators/     69 registered operators (arithmetic, cross-sectional, time-series, group, logic)
├── data/          Synthetic market simulator with planted signals + yfinance loader
├── alphas/        142-formula candidate library across 8 economically motivated families
├── research/      IC, turnover, decay, screening and correlation-based de-duplication
├── ml/            Purged walk-forward CV, ridge and LightGBM signal combination
├── portfolio/     Dollar- and sector-neutral portfolio construction with transaction costs
└── stats/         Deflated Sharpe Ratio and Probability of Backtest Overfitting
```

---

## Quick Start

```bash
# Install
git clone https://github.com/<your-username>/alphalab.git
cd alphalab
pip install -e ".[all]"

# Run the full pipeline (synthetic data, no network needed)
make pipeline

# Run tests
make test

# Run on real prices (needs network)
make pipeline-realdata
```

---

## How It Works

The pipeline runs in seven stages:

### 1. Data Generation

A seeded simulator produces 10 years of daily OHLCV data for 200 instruments with:
- Fat-tailed returns (Student-t with 4 degrees of freedom)
- Volatility clustering (GARCH(1,1))
- Market and sector factor structure
- Three **deliberately planted predictable effects** at realistic strengths:
  - **Short-term reversal**: yesterday's winners tend to give back a little today (MA(1) with negative coefficient)
  - **Momentum**: a persistent AR(1) expected-return component
  - **Volume-conditioned reversal**: reversal is stronger after unusually heavy trading

The planted signals serve as ground truth — if the pipeline cannot recover effects it put there itself, it has a bug.

### 2. Formula Evaluation with Sub-Expression Caching

Alpha formulas are written as strings: `-rank(ts_delta(close, 5)) * rank(volume / ts_mean(volume, 20))`

The engine:
1. **Tokenizes** the string with a hand-written character scanner
2. **Parses** it into a typed AST with a recursive-descent parser (20-rule grammar, right-associative ternary, operator precedence)
3. **Evaluates** the AST against a (dates × instruments) panel, caching every sub-expression by its canonical key

```
142 expressions evaluated in 7.97s
379 of 1271 total nodes are distinct (70.2% reused)
Cache: 228 hits / 607 lookups (37.6% hit rate), 72 entries retained
Without cache: 9.36s → with cache: 7.97s (1.17x speedup)
```

The cache's **admission policy** retains only sub-expressions that appear more than once across the batch. This cut peak memory from 649 MB (naive memoisation) to 125 MB with identical speed.

**Key optimizations that produced measurable gains:**
- Group operators: detecting static sector maps and using column-index slicing instead of per-cell masking → **9–11x faster**
- Rolling argmax/argmin: pre-filling NaN before the sliding view and using prefix-sum NaN detection → **3.5x faster**

### 3. Alpha Measurement

Each of the 142 formulas is scored on:
- **Information Coefficient (IC)**: cross-sectional rank correlation with next-day returns
- **IC-IR**: annualised information ratio of the IC series
- **Turnover**: how much the implied portfolio changes daily
- **Autocorrelation**: how stable the signal is day to day
- **Quantile spread**: return of the top bucket minus the bottom

Top 5 alphas by |IC-IR|:

| Alpha | IC | IC-IR | t-stat | Turnover |
|---|---|---|---|---|
| `sd_rev_simple_1` (sector-neutralised 1-day reversal) | +0.046 | +8.18 | +25.8 | 1.34 |
| `sn_rev_simple_1` (sector-neutral rank of 1-day reversal) | +0.045 | +8.11 | +25.5 | 1.34 |
| `combo_rev_highvolume` (reversal × volume time-series rank) | +0.039 | +6.42 | +20.2 | 1.07 |
| `rev_volume_ranked_1` (ranked reversal × ranked volume) | +0.038 | +6.25 | +19.7 | 1.06 |
| `hump_rev1` (reversal with turnover brake) | +0.043 | +6.12 | +19.3 | 1.33 |

### 4. Screening and De-Duplication

- **Screening**: reject alphas with |t-stat| < 2, implausibly high IC (> 0.25, suspect look-ahead), or turnover > 2.0 → 81 of 142 survive
- **Correlation pruning**: greedily keep the best alphas that are not > 0.7 correlated with any already-kept alpha → **31 survive**

Rejection breakdown: 61 weak, 50 redundant.

### 5. ML Combination (Walk-Forward)

The 31 survivors are combined using **purged walk-forward cross-validation** (6 folds, 1-day purge, 5-day embargo):

- **Equal weight**: average the standardised alphas — the most honest baseline
- **Ridge regression** (α=50): linear combination with strong regularisation
- **LightGBM** (300 trees, 15 leaves, λ=10): can capture alpha interactions

### 6. Backtest Results (at 3 bps one-way cost)

| Model | Sharpe | Total Return | Max Drawdown | Turnover | Names |
|---|---|---|---|---|---|
| Equal weight | 2.17 | 63.1% | −2.8% | 0.47 | 94 |
| **Ridge** | **4.15** | **265.3%** | **−1.9%** | **1.03** | **94** |
| LightGBM | 3.93 | 287.4% | −2.9% | 1.01 | 94 |

**Cost sensitivity** — the most important table in the project:

| Cost assumption | Ridge Sharpe | Interpretation |
|---|---|---|
| 0 bps (gross) | 7.74 | Signal strength before friction |
| 3 bps (institutional) | 4.15 | Large quant fund with internalised crossing |
| 10 bps (retail) | −1.66 | **Costs eat all PnL** — turnover is too high |

The difference between 4.15 and −1.66 is entirely turnover × cost. This is not a flaw in the signal; it is the central trade-off in systematic equity research, and reporting both numbers honestly is the whole point.

### 7. Overfitting Assessment

```
Deflated Sharpe Ratio: 10.581 (p = 0.000)
  Raw Sharpe 4.15, after deflation for 142 trials: 3.21
  Skewness 0.34, excess kurtosis 1.41
Probability of Backtest Overfitting: 0.2% (over 500 combinatorial splits)
```

The raw Sharpe of 4.15 survives the Deflated Sharpe correction (p ≈ 0, haircut to 3.21), meaning the result is unlikely to be explained by selection bias alone. PBO of 0.2% means the in-sample winner almost always wins out of sample too.

---

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │          ALPHA FORMULAS              │
                     │  "-rank(ts_delta(close,5)) * ..."    │
                     └───────────────┬─────────────────────┘
                                     │ tokenize + parse
                                     ▼
                     ┌─────────────────────────────────────┐
                     │            AST NODES                 │
                     │  Call("rank", [Call("ts_delta",...)])│
                     └───────────────┬─────────────────────┘
                                     │ evaluate (with cache)
                     ┌───────────────┼─────────────────────┐
                     │               ▼                      │
                     │  ┌──────────────────────┐           │
                     │  │ SubexpressionCache   │  <-- shared│
                     │  │ key → DataFrame      │  building  │
                     │  └──────────────────────┘  blocks    │
                     │               │                      │
                     └───────────────┼─────────────────────┘
                                     ▼
                     ┌─────────────────────────────────────┐
                     │       SIGNAL PANELS (142)            │
                     │  dates × instruments DataFrames      │
                     └───────────────┬─────────────────────┘
                                     │ measure IC, screen, prune
                                     ▼
                     ┌─────────────────────────────────────┐
                     │       SURVIVORS (31)                 │
                     └───────────────┬─────────────────────┘
                                     │ purged walk-forward ML
                                     ▼
                     ┌─────────────────────────────────────┐
                     │       COMBINED FORECAST              │
                     └───────────────┬─────────────────────┘
                                     │ portfolio construction
                                     ▼
                     ┌─────────────────────────────────────┐
                     │  MARKET-NEUTRAL PORTFOLIO + PnL      │
                     └───────────────┬─────────────────────┘
                                     │ Deflated Sharpe + PBO
                                     ▼
                     ┌─────────────────────────────────────┐
                     │       OVERFITTING REPORT             │
                     └─────────────────────────────────────┘
```

---

## Design Decisions

### Why write a parser instead of using `eval()`?

Formulas are configuration data, not code. `eval()` would let a malicious formula run arbitrary Python. The hand-written parser (tokenizer.py: 120 lines, parser.py: 160 lines) restricts the language to exactly what the evaluator supports, makes error messages point at the exact character that broke, and has no dependency on any parser framework.

### Why cache sub-expressions, and how is it correct?

The 142 formulas share building blocks: `ts_delta(close, 5)` appears in many reversal formulas. Each AST node has a canonical `.key` property, and structurally identical trees produce identical keys by construction. The cache stores the first evaluation and returns it by reference. This is safe because every operator returns a new object and never mutates its input — a property asserted in the test suite.

### Why is the admission policy needed?

Naive memoisation cached every node and held 649 MB. But 70% of distinct sub-expressions appear exactly once, so storing them wasted memory without any speedup. The admission policy pre-scans the batch, counts how often each key appears, and retains only the shared ones. Result: 125 MB, same speed.

### Why purge and embargo in cross-validation?

Standard k-fold is invalid for time series. With a 1-day forward return as the label, the last training label overlaps the first test day. The purge removes training observations whose labels would reach into the test window. The embargo adds a further gap to break serial correlation across the boundary. `tests/test_splits.py` asserts the leakage invariant directly.

### Why does the signal lose money at 10 bps cost?

The dominant alpha family is 1-day reversal, which rebuilds the portfolio almost completely every day (turnover 1.0+). At 10 bps per side, that is 20 bps round-trip daily — roughly 50% annualised cost on a 100% gross book. The gross signal earns about 32% per year, so costs at this level exceed the entire gross return. This is not a failure of the model; it is the fundamental constraint of short-horizon equity strategies.

---

## Caveats (stated, not hidden)

- **Survivorship bias.** The simulated universe does not have delistings. On real data (yfinance mode), the universe is built from today's index members, which overstates historical returns.
- **Simulated data.** The default results are from a simulator. The planted effects are realistic in kind and magnitude, but no simulator captures the full complexity of real markets.
- **No execution model.** Costs are a flat rate on turnover. Real costs include market impact, borrow costs, funding, and queue priority.
- **No fundamental data.** The alpha library uses only price and volume. Real research adds earnings, analyst estimates, news sentiment and alternative data.

---

## Project Structure

```
alphalab/
├── src/alphalab/
│   ├── expr/                  # The formula language
│   │   ├── tokenizer.py       # Hand-written character scanner
│   │   ├── ast_nodes.py       # Typed AST with canonical keys
│   │   ├── parser.py          # Recursive-descent parser
│   │   ├── evaluator.py       # Cached panel evaluator
│   │   ├── cache.py           # LRU cache with admission policy
│   │   └── operators/         # 69 operators in 5 families
│   ├── data/
│   │   ├── synthetic.py       # Market simulator with planted effects
│   │   ├── loaders.py         # Synthetic / yfinance / CSV loader
│   │   └── panel.py           # The (dates × instruments) data structure
│   ├── alphas/
│   │   └── library.py         # 142 formulas in 8 families
│   ├── research/
│   │   ├── metrics.py         # IC, turnover, decay, quantile returns
│   │   └── selection.py       # Screening + correlation pruning
│   ├── ml/
│   │   ├── splits.py          # Purged walk-forward CV
│   │   └── combiner.py        # Ridge / LightGBM / equal-weight
│   ├── portfolio/
│   │   └── construction.py    # Market-neutral portfolio + backtest
│   ├── stats/
│   │   └── overfitting.py     # Deflated Sharpe + PBO
│   └── pipeline.py            # End-to-end runner
├── tests/                     # 58 tests (parser, operators, cache, splits, pipeline)
├── artifacts/                 # Pipeline output (git-ignored, reproducible)
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Technologies

**Core:** Python 3.10+, NumPy, pandas, SciPy, scikit-learn, LightGBM, PyArrow

**Why each is here:**
- NumPy/pandas: vectorised panel operations are 100x faster than row-by-row Python
- SciPy: Student-t distributions in the simulator, skewness/kurtosis for the Deflated Sharpe
- scikit-learn: Ridge regression (one line, no tuning, honest baseline)
- LightGBM: gradient boosting that can capture alpha interactions, if any exist
- PyArrow: Parquet I/O for the data cache

---

## Running with Docker

```bash
docker build -t alphalab .
docker run --rm alphalab
```

---

## License

MIT
