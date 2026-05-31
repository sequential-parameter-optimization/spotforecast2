# Model/Method Card: spotforecast2

This card follows the Hugging Face Model Card Guidebook taxonomy ([Ozoani et al., 2022](https://huggingface.co/docs/hub/model-card-guidebook)). It is the counterpart to the [`spotforecast2-safe` model card](https://github.com/sequential-parameter-optimization/spotforecast2-safe/blob/main/MODEL_CARD.md): where that card documents the deterministic, fail-safe core, this one documents the un-restricted superset built on top of it.

## 1. Model Details

- **Name**: spotforecast2
- **Version**: 3.1.0
- **Type**: Full-featured forecasting toolkit — recursive multi-step forecasting plus hyperparameter tuning (Optuna, SpotOptim), interactive visualization (Plotly), and feature attribution via SHapley Additive exPlanations (SHAP). Un-restricted superset of `spotforecast2-safe`.
- **Developed by**: Thomas Bartz-Beielstein. ORCID: [0000-0002-5938-5158](https://orcid.org/0000-0002-5938-5158).
- **Shared by**: `sequential-parameter-optimization` GitHub organization.
- **Language**: Python 3.13+.
- **License**: AGPL-3.0-or-later.
- **Built on**: `spotforecast2-safe` (`>=15.0.0rc1,<16`) — the safety-critical core, imported and re-exported as `spotforecast2_safe`. The deterministic forecasters, preprocessing, configuration, and data loaders live there.
- **Added capabilities & their dependencies**: visualization — `plotly`, `matplotlib`, `kaleido`; hyperparameter search — `optuna`, `spotoptim`; explainability — `shap`; ENTSO-E data access — `entsoe-py`. Several of these (Plotly, Matplotlib, Optuna, SpotOptim) are exactly the libraries the safe package lists as *prohibited*; `spotforecast2` is where they are allowed to live.
- **Repository**: <https://github.com/sequential-parameter-optimization/spotforecast2>
- **Documentation**: <https://sequential-parameter-optimization.github.io/spotforecast2/>
- **Technical report**: The long-form design rationale and compliance mapping live in the safe package's report (`bart26h/index.qmd` in `spotforecast2-safe`); see also the rendered documentation site above.
- **CPE Identifier (Wildcard)**: `cpe:2.3:a:sequential_parameter_optimization:spotforecast2:*:*:*:*:*:*:*:*`
- **CPE Identifier (Current Release)**: `cpe:2.3:a:sequential_parameter_optimization:spotforecast2:3.1.0:*:*:*:*:*:*:*`

These Common Platform Enumeration (CPE) strings feed Software Bill of Materials (SBOM) and vulnerability-tracking pipelines. Unlike the safe package, this repository ships no CPE-generating utility; the strings above are authoritative and maintained by hand.

## 2. Uses

### Direct Use

- **Exploratory and applied forecasting**: End-to-end electricity load and spot-price forecasting on ENTSO-E data, via the `spotforecast2-entsoe` and `spotforecast-demo` console scripts.
- **Hyperparameter search**: Bayesian search over lags, window features, and regressor parameters using Optuna (`bayesian_search_forecaster`), or surrogate-model search using SpotOptim (`spotoptim_search_forecaster`).
- **Model inspection and explainability**: Interactive Plotly figures (actual-vs-predicted, outliers, distributions, periodograms) and global SHAP feature importances via `shap.TreeExplainer`.
- **Multi-target pipelines**: The `MultiTask` dispatcher and `run()` entry point orchestrate per-target data preparation, outlier handling, imputation, feature engineering, tuning, and prediction.

### Downstream Use

- Selecting lag windows and regressor hyperparameters during development, then **promoting the validated configuration into a `spotforecast2-safe` deployment** for the deterministic inference path.
- Feeding tuned `ForecasterRecursiveLGBMFull` / `ForecasterRecursiveXGBFull` models, or their best parameters, into research notebooks and reporting.

### Out-of-Scope Use

- **Safety-critical inference path**: `spotforecast2` is deliberately *not* the certified, fail-safe layer. It introduces stochastic search, heavier third-party dependencies, and plotting backends. For high-risk, auditable, bit-level-reproducible deployment, use `spotforecast2-safe`.
- **Automated decision-making from plots**: The visualization layer is for human inspection, not for driving automated control loops.
- **Unbounded tuning in production**: Search budgets (`n_trials`, surrogate evaluations) must be bounded; an open-ended search is not an inference contract.

## 3. Bias, Risks, and Limitations

- **Overfitting through tuning**: Optimizing lags and hyperparameters against a limited number of backtest folds can select a configuration that fits the validation window rather than the underlying process. Held-out evaluation is mandatory before a configuration is trusted.
- **Stochasticity and non-determinism**: Optuna's sampler, the SpotOptim surrogate search, and SHAP's subsampling are stochastic. A fixed `random_state` makes a run reproducible, but changing the search budget, the data window, or a dependency version can change the selected model. This is the deliberate opposite of the safe core's bit-level determinism.
- **Approximate explanations**: `shap.TreeExplainer` values are an attribution approximation computed on a sampled fraction (`frac`) of the training data; they indicate, but do not prove, feature relevance.
- **Larger attack surface**: Adding Plotly, Optuna, SpotOptim, SHAP, and their transitive dependencies enlarges the Common Vulnerabilities and Exposures (CVE) surface relative to the minimal safe core. Track the dependency tree in your SBOM.
- **Inherited limitations**: All feature-engineering caveats of the safe core — downstream regressor drift, lag-feature leakage when bypassing the provided builders, and large-series memory cost — apply unchanged here.

### Recommendations

- Validate every tuned configuration against historical ground truth on a held-out horizon before deployment.
- Pin `random_state` and record exact dependency versions to make a tuning run reproducible.
- Bound the search budget (`n_trials`, surrogate evaluations) explicitly.
- Route the production inference path through `spotforecast2-safe`; treat `spotforecast2` as the development and model-selection environment.
- Read SHAP importances as indicative, and corroborate them with backtesting before acting on them.

## 4. How to Get Started

```bash
pip install spotforecast2
```

Tune a LightGBM forecaster with Optuna, then inspect feature importances with SHAP:

```python
from spotforecast2.models import ForecasterRecursiveLGBMFull

# iteration indexes the training run; n_trials bounds the Optuna budget.
model = ForecasterRecursiveLGBMFull(iteration=0, n_trials=10)
model.tune()  # load data, Bayesian search, refit with best params, auto-persist

importances = model.get_global_shap_feature_importance(frac=0.1)
print(importances.head())
```

Run a complete multi-target pipeline programmatically via `run()`, or use the bundled console scripts:

```bash
spotforecast2-entsoe      # ENTSO-E download / train / predict (needs ENTSOE_API_KEY)
spotforecast-demo         # baseline vs. covariate vs. custom-LightGBM comparison
```

Additional N-to-1 pipeline variants are registered as `spotforecast-n2o1`, `spotforecast-n2o1-df`, `spotforecast-n2o1-cov`, and `spotforecast-n2o1-cov-df`.

## 5. Technical Specifications

`spotforecast2` builds on the deterministic sliding-window (N-to-1) lag transformation provided by `spotforecast2-safe` and adds three capability layers on top of it: hyperparameter search, feature attribution, and visualization.

### Hyperparameter search

Two interchangeable engines optimize the forecaster over a search space of candidate lags (`LAGS_CONSIDER`), rolling window features (`WINDOW_FEATURES`: mean / min / max over 24 h, 168 h, and 720 h), and regressor parameters (`SEARCH_SPACES`):

- **Bayesian search** (`bayesian_search_forecaster`) wraps Optuna's Tree-structured Parzen Estimator (TPE) sampler.
- **Surrogate search** (`spotoptim_search_forecaster`) uses SpotOptim, fitting a surrogate model (e.g. a Gaussian process or random forest) to propose configurations with fewer evaluations.

Both are scored by `backtesting_forecaster` over the time-aware folds (`TimeSeriesFold`, `OneStepAheadFold`) re-exported from the safe core, so the evaluation protocol is identical to the deterministic library — only the search around it is added.

### Feature attribution

`get_global_shap_feature_importance(frac)` runs `shap.TreeExplainer` on the fitted tree estimator over a sampled fraction of the training matrix and returns mean absolute SHAP values per feature, sorted descending. An untuned model returns an empty series rather than failing.

### Visualization

`PredictionFigure` and the helpers in `plots/` render actual-vs-predicted traces, outlier overlays, distributions, and periodograms as interactive Plotly figures; `kaleido` provides static image export.

### Architecture (layered)

`forecaster/` (estimator wrappers + metrics, re-exported from the safe core) → `preprocessing/` (outlier detection and ported transformers) → `model_selection/` (`grid_search`, `random_search`, `bayesian_search`, `spotoptim_search`) → `models/` (the `Full` forecasters `ForecasterRecursiveModelFull`, `ForecasterRecursiveLGBMFull`, `ForecasterRecursiveXGBFull`, which override the safe-package `tune()` / `get_global_shap_feature_importance()` stubs) → `multitask/` (the `BaseTask` hierarchy and the `MultiTask` dispatcher) → `plots/` (Plotly visualization) → `tasks/` (console-script entry points).

### Design Objectives

- **Extends, never shadows**: `spotforecast2` overrides only the explicit extension points (`tune`, `get_global_shap_feature_importance`) of the safe core. It does not re-implement deterministic logic with a permissive variant.
- **Reproducible, not deterministic**: a search is repeatable given a fixed `random_state`, dependency set, and data window — a weaker guarantee than the safe core's bit-level determinism, and an intentional one.

## 6. Evaluation

Two evaluation targets apply: forecasting accuracy (the tuned models) and software quality (the toolkit).

### Testing Data

- Docstring examples in `src/` (executable, exercised by the test suite).
- Unit and integration fixtures under `tests/`.
- The bundled ENTSO-E demo dataset and the `~/spotforecast2_data/` test data used by `spotforecast-demo`.

### Factors

- Search engine (Optuna vs. SpotOptim) and budget (`n_trials`, surrogate evaluations).
- Lag-window and rolling-feature configuration.
- Regressor family (LightGBM vs. XGBoost).
- Target count in the multi-target pipeline.

### Metrics

- Forecasting error from `backtesting_forecaster` over `TimeSeriesFold` / `OneStepAheadFold`.
- Functional correctness of the tuning, attribution, and plotting code (unit tests).
- Reproducibility of a search given a fixed `random_state`.
- Coverage on new code, matching the CI configuration.

Unlike the safe core, `spotforecast2` ships no CPE-generation test; the CPE strings in §1 are maintained by hand.

### Results

- **Tuning**: `ForecasterRecursiveLGBMFull` / `ForecasterRecursiveXGBFull` run a full Optuna or SpotOptim search and refit with the best configuration, auto-persisting the model so the predict-only path (`PredictTask`) loads it without re-tuning.
- **Explainability**: global SHAP importances are produced for fitted tree models; an unfitted or untuned model returns an empty series rather than raising.
- **Cybersecurity footprint**: deliberately larger than the safe core. The added visualization, tuning, and explainability stacks are the trade-off for interactive, exploratory use, and are the reason this package is not the safety-critical layer.

## 7. Environmental Impact

Unlike the safe core — which performs no training — `spotforecast2` runs many model fits during a single tuning session: one fit per Optuna trial or surrogate evaluation, multiplied across folds, targets, and iterations. Compute cost, and hence energy use, scales directly with the search budget. The work remains CPU-only (no GPU is required) and no pretrained weights are shipped, but a large `n_trials` over many targets is materially more expensive than a single deterministic forecast. Bound the search budget to control cost.

## 8. Compliance & EU AI Act Support

`spotforecast2` is **not** the path to compliance for a high-risk AI system — that role belongs to `spotforecast2-safe`. The intended division of labor under the EU AI Act (Regulation (EU) 2024/1689) is:

- Use `spotforecast2` during **development**: explore data, search hyperparameters, visualize candidates, and attribute feature importance.
- **Promote the validated configuration** (lags, window features, regressor parameters) into a `spotforecast2-safe` deployment, which provides the deterministic, fail-safe, auditable inference path.

For the authoritative article-by-article mapping (Art. 10 data governance, Art. 11 technical documentation, Art. 12 logging, Art. 13 transparency, Art. 15 accuracy and robustness) to IEC 61508, ISO 26262, ISA/IEC 62443, and the EU AI Act, consult the `spotforecast2-safe` model card and its technical report (`bart26h/index.qmd`).

Within its own scope, `spotforecast2` keeps the transparency properties of the safe core: the code is white-box (no compiled inference kernels, no opaque weights), docstrings are executable, and SHAP attributions are open and inspectable. The stochastic tuning, plotting backends, and enlarged dependency surface are precisely what place this package *outside* the safe envelope by design.

## 9. Glossary

- **AGPL** — Affero General Public License; copyleft license requiring source availability even for network-deployed use.
- **CPE** — Common Platform Enumeration; standardized identifier for software products in vulnerability-tracking systems.
- **CVE** — Common Vulnerabilities and Exposures; public catalogue of known software vulnerabilities.
- **EU AI Act** — Regulation (EU) 2024/1689 on artificial intelligence, in force since 2024-08-01.
- **Optuna** — hyperparameter-optimization framework; here used through its TPE sampler.
- **SBOM** — Software Bill of Materials; machine-readable inventory of a product's components.
- **SHAP** — SHapley Additive exPlanations; game-theoretic method for attributing a model output to its input features.
- **SpotOptim** — surrogate-model-based optimizer (Sequential Parameter Optimization) used as an alternative to Optuna for hyperparameter search.
- **TPE** — Tree-structured Parzen Estimator; the Bayesian sampler Optuna uses by default.
- **IEC 61508 / ISO 26262 / ISA·IEC 62443** — functional-safety and industrial-security standards relevant to the `spotforecast2-safe` compliance path, not to this package. See the safe-package model card.

## 10. Citation

```bibtex
@misc{spotforecast2,
  author       = {Bartz-Beielstein, Thomas},
  title        = {{spotforecast2}: Forecasting Toolkit with Tuning, Visualization, and Explainability},
  year         = {2026},
  howpublished = {\url{https://github.com/sequential-parameter-optimization/spotforecast2}},
  note         = {AGPL-3.0-or-later}
}
```

**APA**: Bartz-Beielstein, T. (2026). *spotforecast2: Forecasting toolkit with tuning, visualization, and explainability* (Version 3.1.0) [Computer software]. https://github.com/sequential-parameter-optimization/spotforecast2

The long-form design rationale, compliance mapping, and evaluation protocol live in the `spotforecast2-safe` technical report (`bart26h/index.qmd`).

## 11. Model Card Authors & Contact

- Thomas Bartz-Beielstein — ORCID [0000-0002-5938-5158](https://orcid.org/0000-0002-5938-5158) — `bartzbeielstein@gmail.com`

This card follows the Hugging Face Model Card Guidebook taxonomy ([Ozoani et al., 2022](https://huggingface.co/docs/hub/model-card-guidebook)).

## 12. How to Audit

`spotforecast2` *intentionally* ships the libraries that `spotforecast2-safe` prohibits, so the audit goal is not dependency minimization but correct separation of concerns:

1. Confirm that `spotforecast2` is used for development and model selection, and that the safety-critical inference path runs on `spotforecast2-safe` — not on this package.
2. Run `uv run pytest tests/` to verify the tuning, attribution, and visualization code.
3. Check that tuning runs pin `random_state` (and record dependency versions) wherever reproducibility is required.
4. Reference the CPE identifiers from §1 in vulnerability-tracking systems and SBOM disclosures. Note that, unlike the safe package, this repository ships no CPE-generating utility or test — the strings in §1 are authoritative.
5. Run `uv tool run reuse lint` to confirm SPDX / REUSE licensing compliance.

## 13. Disclaimer & Liability

**LIMITATION OF LIABILITY**: This software is provided "AS IS" without any warranties. The developers and contributors assume **NO LIABILITY** for any direct or indirect damages, system failures, or financial losses resulting from its use.

`spotforecast2` is an exploratory and model-selection toolkit and is **not intended for safety-critical deployment**. For production or safety-critical use, deploy `spotforecast2-safe` and perform a full system-level safety validation (e.g., as per ISO 26262, IEC 61508, or the EU AI Act) before going live.
