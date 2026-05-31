# Model/Method Card: spotforecast2

This card describes what spotforecast2 is, how to use it, the conditions under which its results are valid, and how it relates to its safety-critical sibling spotforecast2-safe. It follows the Hugging Face Model Card Guidebook taxonomy ([Ozoani et al., 2022](https://huggingface.co/docs/hub/model-card-guidebook)). Where the [`spotforecast2-safe` model card](https://github.com/sequential-parameter-optimization/spotforecast2-safe/blob/main/MODEL_CARD.md) documents the deterministic, fail-safe core, this card documents the un-restricted superset built on top of it.

## 1. Model Details

| Field | Value |
| --- | --- |
| Name | spotforecast2 |
| Version | 3.3.0 |
| Type | Full-featured forecasting toolkit: recursive multi-step forecasting plus hyperparameter tuning (Optuna, SpotOptim), interactive visualization (Plotly), and feature attribution via SHapley Additive exPlanations (SHAP). Un-restricted superset of `spotforecast2-safe`. |
| Developed by | Thomas Bartz-Beielstein, ORCID [0000-0002-5938-5158](https://orcid.org/0000-0002-5938-5158) |
| Distributed by | the `sequential-parameter-optimization` GitHub organization |
| Language | Python 3.13 or newer |
| License | AGPL-3.0-or-later (Affero General Public License) |
| Built on | `spotforecast2-safe` (`>=15.0.0rc1,<16`), imported and re-exported as `spotforecast2_safe` |
| Repository | <https://github.com/sequential-parameter-optimization/spotforecast2> |
| Documentation | <https://sequential-parameter-optimization.github.io/spotforecast2/> |
| Technical report | the safe package's `bart26h/index.qmd`, plus the rendered documentation above |

spotforecast2 inherits the deterministic forecasters, preprocessing, configuration, and data loaders from spotforecast2-safe and adds four capability groups that the safe package deliberately excludes: visualization (plotly, matplotlib, kaleido), hyperparameter search (optuna, spotoptim), explainability (shap), and ENTSO-E data access (entsoe-py). Several of these — Plotly, Matplotlib, Optuna, SpotOptim — are exactly the libraries the safe package lists as prohibited; spotforecast2 is where they are allowed to live.

Two Common Platform Enumeration (CPE) identifiers let vulnerability-tracking and software bill of materials (SBOM) tools recognize the package. The wildcard identifier `cpe:2.3:a:sequential_parameter_optimization:spotforecast2:*:*:*:*:*:*:*:*` matches any release; the current release is `cpe:2.3:a:sequential_parameter_optimization:spotforecast2:3.3.0:*:*:*:*:*:*:*`. Unlike the safe package, this repository ships no CPE-generating utility; the strings above are authoritative and maintained by hand.

spotforecast2 is, by design, a development and model-selection tool rather than a low-risk deterministic component. It introduces stochastic search, plotting backends, and a larger dependency tree, which place it outside the safe envelope. It is not the certified inference path for a high-risk AI system; that role belongs to spotforecast2-safe. Responsibilities are divided as follows.

| Responsibility | Party | Contact |
| --- | --- | --- |
| Toolkit development and maintenance | Thomas Bartz-Beielstein | `bartzbeielstein@gmail.com` |
| Distribution | sequential-parameter-optimization on GitHub | repository issue tracker |
| Model selection and experimentation | the analyst using this package | per project |
| Safety-critical deployment, operation, and audit | the system integrator, through `spotforecast2-safe` | defined per deployment |

The current release is 3.3.0, tracking the stable public interface re-exported alongside `spotforecast2_safe`. The full version history, including release dates, is recorded in `CHANGELOG.md` and on the GitHub Releases page; it is maintained automatically by the release pipeline and is not repeated here.

## 2. Intended Use and Scope

spotforecast2 supports exploratory and applied forecasting of hourly electricity load and spot prices on ENTSO-E data, end to end, through the `spotforecast2-entsoe` and `spotforecast-demo` console scripts. Its distinctive capabilities are hyperparameter search — Bayesian search over lags, window features, and regressor parameters with Optuna (`bayesian_search_forecaster`), or surrogate-model search with SpotOptim (`spotoptim_search_forecaster`) — interactive inspection through Plotly figures, and global SHAP feature importances via `shap.TreeExplainer`. The `MultiTask` dispatcher and the `run()` entry point orchestrate multi-target pipelines: per-target data preparation, outlier handling, imputation, feature engineering, tuning, and prediction.

The intended downstream use is development and model selection: choosing lag windows and regressor hyperparameters here, then promoting the validated configuration into a spotforecast2-safe deployment for the deterministic inference path. Tuned `ForecasterRecursiveLGBMFull` and `ForecasterRecursiveXGBFull` models, or just their best parameters, also feed research notebooks and reporting.

The scope has firm limits. spotforecast2 is deliberately not the certified, fail-safe inference layer: it introduces stochastic search, heavier third-party dependencies, and plotting backends. For high-risk, auditable, bit-for-bit reproducible deployment, use spotforecast2-safe. The visualization layer is for human inspection, not for driving automated control loops. Search budgets (`n_trials`, surrogate evaluations) must be bounded; an open-ended search is not an inference contract.

## 3. How to Get Started

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

## 4. Technical Specification

### Task and model family

spotforecast2 addresses recursive multi-step forecasting of a univariate series from its own lags, rolling-window features, and exogenous regressors, using the same scikit-learn-compatible wrappers as the safe core. On top of that deterministic base it adds three capability layers: hyperparameter search, feature attribution, and visualization.

### Lag construction

The sliding-window (N-to-1) lag transformation, which builds one feature row per target value without ever placing the target in its own feature row, is inherited unchanged from spotforecast2-safe; see that package's model card for the mathematical description. spotforecast2 does not re-implement it.

### Hyperparameter search

Two interchangeable engines optimize the forecaster over a search space of candidate lags (`LAGS_CONSIDER`), rolling window features (`WINDOW_FEATURES`: mean, min, and max over 24 h, 168 h, and 720 h), and regressor parameters (`SEARCH_SPACES`):

- **Bayesian search** (`bayesian_search_forecaster`) wraps Optuna's Tree-structured Parzen Estimator (TPE) sampler.
- **Surrogate search** (`spotoptim_search_forecaster`) uses SpotOptim, fitting a surrogate model to propose configurations with fewer evaluations.

Both are scored by `backtesting_forecaster` over the time-aware folds (`TimeSeriesFold`, `OneStepAheadFold`) re-exported from the safe core, so the evaluation protocol is identical to the deterministic library — only the search around it is added.

### Feature attribution

`get_global_shap_feature_importance(frac)` runs `shap.TreeExplainer` on the fitted tree estimator over a sampled fraction of the training matrix and returns mean absolute SHAP values per feature, sorted descending. An untuned model returns an empty series rather than failing.

### Visualization

`PredictionFigure` and the helpers in `plots/` render actual-vs-predicted traces, outlier overlays, distributions, and periodograms as interactive Plotly figures; `kaleido` provides static image export.

### Training

Unlike the safe core, which trains nothing on its own, spotforecast2 performs real training during a tuning session. The `Full` forecasters override the safe-package `tune()` stub with an Optuna or SpotOptim search and refit with the best configuration, then auto-persist the model so the predict-only path (`PredictTask`) loads it without re-tuning. A single session runs many fits — one per Optuna trial or surrogate evaluation, multiplied across folds, targets, and iterations — so the training cost is a function of the search budget rather than a fixed quantity.

### Architecture (layered)

`forecaster/` (estimator wrappers plus metrics, re-exported from the safe core) → `preprocessing/` (outlier detection and ported transformers) → `model_selection/` (`grid_search`, `random_search`, `bayesian_search`, `spotoptim_search`) → `models/` (the `Full` forecasters `ForecasterRecursiveModelFull`, `ForecasterRecursiveLGBMFull`, `ForecasterRecursiveXGBFull`, which override the safe-package `tune()` and `get_global_shap_feature_importance()` stubs) → `multitask/` (the `BaseTask` hierarchy and the `MultiTask` dispatcher) → `plots/` (Plotly visualization) → `tasks/` (console-script entry points).

### Design objectives

- **Extends, never shadows**: spotforecast2 overrides only the explicit extension points (`tune`, `get_global_shap_feature_importance`) of the safe core. It does not re-implement deterministic logic with a permissive variant.
- **Reproducible, not deterministic**: a search is repeatable given a fixed `random_state`, dependency set, and data window — a weaker guarantee than the safe core's bit-for-bit determinism, and an intentional one.

## 5. Interfaces and Runtime

The input and output contract is inherited from the safe core. The target is a numeric univariate series on a regular, monotonic date-time index; exogenous features are a complete numeric frame aligned to that index, and any missing exogenous entry raises a `ValueError` before a prediction is made. Forecasts are returned as a series whose length equals the requested horizon, in the target's units. Fitted forecasters are serialized with joblib and reloaded through the same persistence helpers, so a tuned model produced here loads unchanged in a spotforecast2-safe predict path.

spotforecast2 runs on Python 3.13 or newer on a central processing unit (CPU); it needs no graphics processing unit (GPU) and ships no GPU code. No pretrained weights are shipped.

The added capabilities bring their own dependencies. All but one carry permissive licenses; SpotOptim is copyleft (AGPL-3.0-or-later), consistent with spotforecast2's own license but a difference from the safe core's all-permissive runtime.

| Dependency | Capability | License |
| --- | --- | --- |
| plotly | interactive visualization | MIT |
| matplotlib | static visualization | Python Software Foundation |
| kaleido | static image export | MIT |
| optuna | Bayesian hyperparameter search | MIT |
| spotoptim | surrogate hyperparameter search | AGPL-3.0-or-later |
| shap | feature attribution | MIT |
| entsoe-py | ENTSO-E data access | MIT |

These additions enlarge the Common Vulnerabilities and Exposures (CVE) attack surface relative to the minimal safe core; track the full dependency tree, including transitive dependencies, in your SBOM. Compute cost is dominated by the tuning search, not by inference: a single forecast is cheap, but a large `n_trials` over many targets runs proportionally many model fits and is materially more expensive than one deterministic forecast. The work stays CPU-only; bound the search budget to control cost and energy use.

## 6. Data and Operational Design Domain

The Operational Design Domain (ODD) is the set of conditions under which the toolkit's results are valid. spotforecast2 inherits the safe core's data ODD and adds conditions specific to tuning and attribution. Outside these conditions the inherited layers raise an error rather than return an unreliable result; the added search and attribution layers instead lose trustworthiness, which is why the recommendations below matter.

| Condition | Valid range | Outside the range |
| --- | --- | --- |
| Target series | numeric, univariate, regular monotonic date-time index | error (inherited) |
| Exogenous features | numeric, complete, aligned to the target index | `ValueError` on any missing entry (inherited) |
| Sampling interval | uniform; hourly in the demos | unreliable result |
| Minimum history | longer than the window size plus the number of lags | the model cannot be called |
| Search budget | bounded `n_trials` / surrogate evaluations | unbounded search is not an inference contract |
| Reproducibility | fixed `random_state`, pinned dependency versions and data window | the selected model may change between runs |
| Tuning validation | held-out ground truth beyond the backtest folds | overfitting to the validation window |
| SHAP attribution | computed on a sampled fraction (`frac`) of training data | values indicate, but do not prove, feature relevance |

Two risks follow from the added layers. Optimizing lags and hyperparameters against a limited number of backtest folds can select a configuration that fits the validation window rather than the underlying process, so held-out evaluation is mandatory before a configuration is trusted. And because Optuna's sampler, the SpotOptim surrogate search, and SHAP's subsampling are stochastic, a fixed `random_state` makes a run reproducible, but changing the search budget, the data window, or a dependency version can change the selected model — the deliberate opposite of the safe core's bit-for-bit determinism. All feature-engineering caveats of the safe core — regressor drift, leakage when bypassing the provided builders, and large-series memory cost — apply here unchanged.

To stay inside the valid domain: validate every tuned configuration against historical ground truth on a held-out horizon, pin `random_state` and record exact dependency versions, bound the search budget explicitly, and route the production inference path through spotforecast2-safe.

## 7. Evaluation

Two evaluation targets apply: forecasting accuracy (the tuned models) and software quality (the toolkit). Tuning is scored by `backtesting_forecaster` over `TimeSeriesFold` and `OneStepAheadFold`, so a configuration's forecasting error is measured on time-aware folds that never leak future data into a past prediction. The `Full` forecasters run a complete Optuna or SpotOptim search and refit with the best configuration, auto-persisting the result so the predict-only path loads it without re-tuning; global SHAP importances are produced for fitted tree models, and an unfitted or untuned model returns an empty series rather than raising.

The toolkit itself is evaluated on software-quality properties: docstring examples in `src/` are executed by the test suite, unit and integration fixtures live under `tests/`, reproducibility of a search is checked against a fixed `random_state`, and new code carries coverage matching the CI configuration. Unlike the safe core, spotforecast2 ships no CPE-generation test — the CPE strings in Section 1 are maintained by hand — and its cybersecurity footprint is deliberately larger, the trade-off for interactive, exploratory use.

## 8. Model Transparency

spotforecast2 produces point forecasts. It does not natively quantify or calibrate predictive uncertainty, so a deployment that needs prediction intervals or calibrated probabilities must add them in the downstream regressor or a wrapper around it — a limitation inherited from the safe core.

Where the safe package deliberately ships no explainability backend, spotforecast2 adds one: `get_global_shap_feature_importance` exposes global SHAP attributions from `shap.TreeExplainer` on top of the regressor's own split- and gain-based importances. The code remains white-box — no compiled inference kernels, no opaque weights — and the docstrings are executable, so every transformation and every attribution can be read and audited in source.

## 9. Operation: Monitoring and Response

A deployment that uses spotforecast2 during development should watch the quality of incoming data (missing or out-of-range values and timestamp gaps), the drift of the input and target distributions away from the tuning period, and the forecast error against a simple seasonal baseline. The ENTSO-E task enforces a retraining-cadence gate through the configuration's `retrain_max_age`: a model older than the configured age triggers a retrain, and the `--force` flag bypasses the gate when an immediate refit is wanted. A module-level logger records timestamped events for audit retention.

When monitoring crosses a deployment-defined threshold, the usual responses are to retune or retrain, to fall back to the seasonal baseline or the last known-good model, and to alert the responsible team. Once a validated configuration is promoted to a spotforecast2-safe deployment, the integrator owns the thresholds and escalation steps for the safety-critical inference path.

## 10. Compliance Support

spotforecast2 is not the path to compliance for a high-risk AI system — that role belongs to spotforecast2-safe. The intended division of labor under the EU AI Act (Regulation (EU) 2024/1689) is to use spotforecast2 during development — explore data, search hyperparameters, visualize candidates, and attribute feature importance — and then promote the validated configuration (lags, window features, regressor parameters) into a spotforecast2-safe deployment, which provides the deterministic, fail-safe, auditable inference path.

For the authoritative article-by-article mapping (Article 10 data governance, Article 11 technical documentation, Article 12 logging, Article 13 transparency, Article 15 accuracy and robustness) to IEC 61508, ISO 26262, and ISA/IEC 62443, consult the spotforecast2-safe model card and its technical report (`bart26h/index.qmd`). Within its own scope, spotforecast2 keeps the transparency properties of the safe core: white-box code, executable docstrings, and open, inspectable SHAP attributions. The stochastic tuning, plotting backends, and enlarged dependency surface are precisely what place this package outside the safe envelope by design.

## 11. Glossary

| Term | Meaning |
| --- | --- |
| Optuna | Hyperparameter-optimization framework; here used through its TPE sampler. |
| SpotOptim | Surrogate-model-based optimizer (Sequential Parameter Optimization) used as an alternative to Optuna for hyperparameter search. |
| Surrogate model | A cheap approximation of the expensive objective (here, backtest error) that the optimizer queries to propose new configurations with fewer full evaluations. |
| Backtesting | Evaluation of a forecaster by replaying it over historical, time-aware folds (`TimeSeriesFold`, `OneStepAheadFold`). |
| IEC 61508 / ISO 26262 / ISA·IEC 62443 | Functional-safety and industrial-security standards relevant to the spotforecast2-safe compliance path, not to this package. See the safe-package model card. |

## 12. How to Audit

spotforecast2 intentionally ships the libraries that spotforecast2-safe prohibits, so the audit goal is not dependency minimization but correct separation of concerns.

1. Confirm that spotforecast2 is used for development and model selection, and that the safety-critical inference path runs on spotforecast2-safe — not on this package.
2. Run `uv run pytest tests/` to verify the tuning, attribution, and visualization code.
3. Check that tuning runs pin `random_state` (and record dependency versions) wherever reproducibility is required.
4. Record the CPE identifiers from Section 1 in vulnerability-tracking systems and SBOM disclosures. Unlike the safe package, this repository ships no CPE-generating utility or test — the strings in Section 1 are authoritative.
5. Run `uv tool run reuse lint` to confirm SPDX and REUSE license-header compliance.

## 13. Citation, Authors, and Contact

Maintainer: Thomas Bartz-Beielstein, ORCID [0000-0002-5938-5158](https://orcid.org/0000-0002-5938-5158), `bartzbeielstein@gmail.com`.

```bibtex
@misc{spotforecast2,
  author       = {Bartz-Beielstein, Thomas},
  title        = {{spotforecast2}: Forecasting Toolkit with Tuning, Visualization, and Explainability},
  year         = {2026},
  howpublished = {\url{https://github.com/sequential-parameter-optimization/spotforecast2}},
  note         = {AGPL-3.0-or-later}
}
```

Or as a formatted reference: Bartz-Beielstein, T. (2026). *spotforecast2: Forecasting toolkit with tuning, visualization, and explainability* (Version 3.3.0) [Computer software]. https://github.com/sequential-parameter-optimization/spotforecast2

The long-form design rationale, compliance mapping, and evaluation protocol live in the spotforecast2-safe technical report (`bart26h/index.qmd`).

## 14. Disclaimer and Liability

**Limitation of liability.** This software is provided "AS IS" without any warranties. The developers and contributors assume no liability for any direct or indirect damages, system failures, or financial losses resulting from its use.

spotforecast2 is an exploratory and model-selection toolkit and is not intended for safety-critical deployment. For production or safety-critical use, deploy spotforecast2-safe and perform a full system-level safety validation (for example under ISO 26262, IEC 61508, or the EU AI Act) before going live.
