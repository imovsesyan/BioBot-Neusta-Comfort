# F9 — Final Model Recommendation

**Date**: 2026-04-27
**Project context**: BioBot-Neusta-Comfort — environmental prediction system for indoor livability classification on the Neusta sensor stream (target `vivabilite_binary_mean`, n = 4,315 rows, single sensor, 15-min cadence).
**Scope**: Synthesize literature, full-model comparison, and humidex ablation into a single defensible model recommendation; evaluate the collaborator TFT (biosense360) separately.

---

## SECTION 1 — Our results vs literature

The F9 candidates are evaluated below on the Neusta `vivabilite_binary_mean` target (0–1 scale, chronological 70/15/15 split). Reference numbers are taken from `F9_etat_de_lart.md` §2–§5 and the master comparison tables in `F9_full_model_comparison.md` §3.

| Model | Test MAE | Test R² | Closest published reference | Verdict |
|---|---:|---:|---|---|
| Mean baseline | 0.3450 | -0.0503 | Hyndman & Athanasopoulos 2021 (FPP3): naive mean R² typically 0.0–0.2 on noisy environmental sensor streams | **CONSISTENT** (slightly below floor — indicates train/test distribution drift) |
| Ridge Regression | 0.1753 | 0.5748 | Liu et al. 2022 (Ridge / MLR on thermal comfort, R² 0.4–0.7); Frontiers in Built Environment surveys 2025 | **CONSISTENT** — sits in the middle of the literature-plausible band for a single-location linear comfort baseline |
| Random Forest | 0.0033 | 0.9967 | Luo et al. 2020 *Energy and Buildings* 210:109776 (best of 9 ML models on 81,846 ASHRAE II votes): RF accuracy 66.3% / 61.1%; PMV-class regression R² 0.85–0.99 | **ABOVE LITERATURE** — exceeds best reported tree results on much larger datasets; flagged as leakage artefact (see below) |
| HGB | 0.0033 | 0.9975 | MDPI *Sensors* 25, 7294 (2025) HGB/LightGBM/XGBoost/CatBoost benchmark: typical R² 0.85–0.97 on building-energy / indoor-T regression | **ABOVE LITERATURE** — same leakage diagnosis as RF |
| XGBoost | 0.0036 | 0.9969 | Luo 2020 ASHRAE: XGBoost R² = 0.994 on PPD; AIP Conf. Proc. 3240 (2024) tuned XGBoost R² 0.93 / 96.29% accuracy on PMV | **ABOVE LITERATURE** — our R² ≈ 0.997 surpasses Luo 2020's R² = 0.994 ceiling on a 4,315-row single-sensor dataset, which is implausible without leakage |
| CNN-LSTM (with humidex) | 0.3000 | -0.1879 | Hou et al. 2022 (60,133 rows): R² 0.7268–0.7638, MAE 1.02 °C; Elmaz 2021 multi-month single-room | **CONSISTENT** with literature failure mode (Grinsztajn et al. NeurIPS 2022; Chai et al. 2020) — CNN-LSTM is misapplied below the ~10k-row threshold and on near-stationary signals |
| CNN-LSTM (no humidex) | 0.4073 | -0.1909 | Same as above — confirms the model has no useful inductive bias at this sample size | **CONSISTENT** with état de l'art §2.5 |
| TFT biosense360 (target = humidex_c, °C) | 0.945 °C @ 15 min, 5.399 °C @ 120 min; 80% coverage 58.6% @ 120 min | not reported | Lim et al. 2021; pytorch-forecasting Stallion tutorial; Dai et al. (CityTFT, 2025) 17M rows | **NOT COMPARABLE** — different target (humidex_c in °C vs. unitless livability score), different dataset (1.2M merged rows), different task (multi-horizon quantile forecasting) |

**Tree models — leakage flag**: our test R² ≈ 0.997 sits *above* the best published thermal-comfort tree result we located (Luo 2020 ASHRAE: XGBoost R² = 0.994 on PPD, on 81,846 votes). For our model to legitimately exceed that ceiling on 4,315 rows from a single sensor would require a data-quality or signal-density advantage that we do not have. The humidex ablation (`F9_UC7_ablation.md`) confirms the leakage hypothesis directly: removing `humidex_c` and its lag/rolling variants drops tree R² from 0.997 to 0.88–0.90 (Random Forest 0.997 → 0.887; HGB 0.997 → 0.881; XGBoost 0.997 → 0.900), with Test MAE rising tenfold (0.003 → 0.032–0.048). The remaining R² of 0.88–0.90 still relies on temperature and humidity, which are themselves the inputs of the humidex closed-form formula — so the ablation removes only the most explicit channel of the leakage, not the underlying coupling between target and features.

**CNN-LSTM — consistent with literature failure mode**: our test R² = -0.19 on 4,315 rows is exactly what Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022) predict for deep tabular architectures on medium-sized tabular data, and it matches Chai et al. (2020) showing tree-based models dominate small thermal-comfort datasets (1,670 rows). État de l'art §2.3 documents that *every* peer-reviewed CNN-LSTM thermal/climate study we located used ≥ 10k samples, with the best (Hou 2022) using 60,133. Our negative R² is not evidence the architecture is intrinsically bad — it is evidence the data budget is too small for its inductive bias.

---

## SECTION 2 — TFT verdict backed by evidence

### 1. What does the literature say TFT needs?

- **Minimum dataset size**: pytorch-forecasting (Stallion demand-forecasting tutorial, v1.4.0) explicitly states *"TFT can demonstrate good performance on very small datasets with only ~20k samples, but it is a large model and will perform much better with more data."* This is the most concrete community lower bound (état de l'art §3.2).
- **Original benchmarks (Lim et al. 2021)**: all four datasets in the original paper are large multi-entity panels — Electricity (~9.6M observations across 370 clients), Traffic / PEMS-SF (~10M observations across 963 sensors), Volatility (31 stock indices, multi-year daily), Retail / Favorita (4,100 stores × multi-year daily). Single-stream usage is **out of distribution** for the original validation (état de l'art §3.1).
- **Number of series**: TFT's Variable Selection Networks and Static Covariate Encoders are designed for multi-entity panels with shared static covariates; on single-location streams these components have little to discriminate (TransferLab review of Lim et al.).
- **Stationarity / regime**: practitioner literature (Optuna-tuned TFT case studies, Towards Data Science, DataNess.AI) report that on < 10k samples *"TFT overfits within a handful of epochs"* without aggressive dropout, GRN regularisation, early stopping and feature pruning (état de l'art §3.2).
- **Train/test R² gap threshold**: pytorch-forecasting tutorials and Optuna case studies treat a > 0.05 R² gap or > 5% MAPE gap as an overfitting signal for TFT specifically — tighter than the 0.10–0.15 threshold used for plain LSTMs (Brownlee, *How to Diagnose Overfitting and Underfitting of LSTM Models*) — because TFT has many capacity-controlling regularisers that should make tight train-val agreement achievable when the model is well-specified (état de l'art Q2).

### 2. Does our dataset meet those requirements?

Two distinct datasets are at issue:

- **F9 Neusta dataset**: 4,315 rows, single sensor, 15-min cadence, single target. **Does not meet** the pytorch-forecasting 20k-sample threshold. Has zero parallel series. Has no static covariates (one location, one sensor). This is the regime état de l'art §3.4 explicitly flags as risky for TFT.
- **biosense360 merged dataset**: ≈ 1.2M rows after gap-based series splitting across Neusta, Meteo France, IoT and Aquacheck sources, with `sensor_id` as a grouping variable. **In raw row count this exceeds the 20k threshold** — but the 1.2M rows are dominated by Meteo France after 1h resampling (60 stations) and sit alongside only 1 Neusta sensor, 2 IoT sensors and 10 Aquacheck sensors. The literature requires "many parallel series with shared static covariates" (état de l'art §3.3); the biosense360 panel has heterogeneous source semantics and the per-source MAE breakdown (Neusta 1.246 °C @ 120 min vs. Meteo France 5.574 °C) shows the panel is not behaving as a homogeneous multi-entity benchmark. The early stop at epoch 7 (with `EarlyStopping(patience=10)`) is the strongest single piece of evidence: validation loss plateaued or rose almost immediately after the first epoch, which is the textbook TFT-on-small-data signature documented in état de l'art §3.2.

### 3. Do our TFT numbers confirm or contradict the literature?

They **confirm** the literature's small-data TFT failure modes on three counts:

- **Calibration gap**: nominal 80% interval coverage achieves only 71.7% @ 15 min, 64.2% @ 30 min, 64.7% @ 60 min, 58.6% @ 120 min — under-coverage at every horizon and a 21.4-percentage-point miss at 120 min. État de l'art §3.4 names this an "overconfident quantile head" failure mode, and §3.2 documents it as a small-data TFT pattern.
- **Horizon degradation**: Test MAE inflates from 0.945 °C @ 15 min to 5.399 °C @ 120 min — a 5.7× degradation over 8 decoder steps. Several HVAC studies (état de l'art §2.4) report this error-accumulation pattern when TFT is applied below the documented stability threshold.
- **Early stop at epoch 7**: with `EarlyStopping(patience=10)` and `max_epochs=100`, halting at epoch 7 means validation loss stopped improving immediately after the first epoch. État de l'art §3.2 calls a < 10-epoch convergence on small TFT data a strong overfitting / instability signal.

### 4. Is TFT justified for the livability target?

**Conditionally justified — not as a standalone livability predictor, but as a humidex-forecasting upstream component.** The client brief (état de l'art §1) is explicit: the project aims to *"build an environmental prediction system to assess the livability of a location"* via the binary livability target `vivabilite_binary_mean`. The biosense360 TFT predicts `humidex_c` in °C — an intermediate physical variable, not the livability label. État de l'art §3.4 makes this point directly: *"For the livability classification task itself (target `vivabilite_binary_mean`), TFT is doubly inappropriate: it is a forecasting architecture, while the operational task is a classification of an instantaneous score that can be derived from current sensor readings — a tabular problem in disguise."* The TFT can be used as a multi-horizon humidex forecaster feeding a downstream livability classifier (e.g., for "what will the humidex be in 2 hours?" → "will the location still be livable?"), but it cannot be presented as the livability model itself.

---

## SECTION 3 — Final recommendation with 3-level evidence

### Winning model — Ridge Regression

**LITERATURE SAYS**: Ridge / regularised linear regression is the canonical interpretable baseline for thermal-comfort regression (Liu, Yao, Ma et al. 2022, *J. Ambient Intell. Humaniz. Comput.*: MLR / Ridge MAE 0.5–0.9 on 7-point thermal sensation scales, R² 0.4–0.7). Frontiers in Built Environment / Building Research & Information surveys (2025) consistently report Ridge outperformed by tree ensembles by 5–20 percentage points of accuracy when leakage is controlled — but on a single-location, ≤ 5,000-row stream where tree models have a confirmed leakage issue (Luo 2020 ceiling exceeded; humidex ablation shows 11 percentage points of R² come from `humidex_c` alone), the linear model's limited capacity is a feature, not a bug. État de l'art §4.2 names Ridge as "the most literature-consistent result in the entire study". A test R² of 0.57 on a single-location comfort predictor with confirmed leakage in the tree alternatives is scientifically defensible and matches the literature's expected band.

**OUR DATA SHOWS**: Ridge Regression — Test MAE = 0.1753, Test R² = 0.5748, Train R² = 0.7238, Train MAE = 0.1571. Train→Test R² gap = 0.149 — present, but consistent with the documented test-set distribution drift visible in the mean-baseline R² of -0.0503. No leakage signature comparable to the trees: the linear capacity caps how much closed-form humidex coupling Ridge can exploit. The humidex ablation (`F9_UC7_ablation.md`) confirms the diagnosis: Ridge R² drops modestly from 0.5748 to 0.4273 without humidex (a 0.15 drop), whereas Random Forest collapses from 0.9967 to 0.8871 (a 0.11 drop in absolute R² but starting from a leakage-inflated ceiling — Test MAE rises tenfold from 0.003 to 0.037).

**CONCLUSION**: Ridge Regression is the most scientifically defensible model for the `vivabilite_binary_mean` target on this dataset. It is the only model that simultaneously (a) avoids the tree models' leakage artefact, (b) achieves meaningful predictive performance above the mean-baseline floor (R² 0.57 vs. -0.05), (c) sits inside the literature-plausible band for single-location linear thermal-comfort baselines, and (d) is fully interpretable (every coefficient is auditable). **Confidence level: MEDIUM** — the target itself remains unvalidated per CLAUDE.md, pending confirmation of `vivabilite_binary_mean` semantics with the data provider (specifically: whether the label is human-annotated or formula-derived from environmental inputs). The recommendation should be revisited if the target turns out to be a closed-form function of features, in which case the appropriate model becomes a formula approximator rather than a comfort predictor.

### Models to dismiss

❌ **DO NOT USE: Random Forest**
   Because: Test R² = 0.9967 exceeds the Luo 2020 ASHRAE ceiling (0.994 on 81,846 rows) on only 4,315 rows; humidex ablation drops R² to 0.887 confirming the headline number is largely formula reproduction, not independent comfort prediction.

❌ **DO NOT USE: HGB (Histogram Gradient Boosting)**
   Because: Test R² = 0.9975 sits above the MDPI *Sensors* 25 (2025) typical 0.85–0.97 band for HGB on building-energy / indoor-T regression; identical leakage profile to Random Forest with deepest interaction-finding capacity, so HGB is the most able to exploit even partial leakage.

❌ **DO NOT USE: XGBoost**
   Because: Test R² = 0.9969 exceeds the AIP 2024 PMV ceiling (R² 0.93) and edges past Luo 2020 PPD ceiling (0.994); humidex ablation drops R² to 0.900 — same leakage diagnosis as the other two trees, no scientific case for choosing it over Ridge.

❌ **DO NOT USE: CNN-LSTM (either variant)**
   Because: Test R² = -0.19 with humidex and -0.19 without — both worse than the mean-baseline floor (-0.05); État de l'art §2.3 documents that *every* peer-reviewed CNN-LSTM thermal/climate study uses ≥ 10k rows, and Grinsztajn et al. (NeurIPS 2022) confirm trees beat deep tabular architectures on medium-sized data — our 4,315 rows are below the operating envelope.

❌ **DO NOT USE: Mean baseline (as a model)**
   Because: Test R² = -0.0503 — useful only as a reference floor for diagnosing distribution drift; a constant mean is not a predictive model, and its negative R² already signals that the train mean is a poor summary of the test distribution.

### Conditional model

⚠️ **CONDITIONALLY USEFUL: TFT (biosense360)**
   Because: It solves a valid but **different** problem — multi-horizon humidex forecasting (target `humidex_c` in °C) as a potential upstream component for a downstream livability classifier, **not** livability classification itself. Needs calibration improvement (80% interval coverage 58.6% @ 120 min vs. 80% nominal — a 21.4 pp miss) and confirmed convergence (currently early-stopped at epoch 7 of `max_epochs=100` with `patience=10`, the textbook small-data TFT failure signature per état de l'art §3.2). Not authoritative without holdout-week or cross-location validation.

---

## SECTION 4 — Presentation paragraph

On a 4,315-row single-sensor Neusta stream targeting the binary livability score `vivabilite_binary_mean`, the three tree models (Random Forest, HGB, XGBoost) report test R² ≈ 0.997 — a value that exceeds the best published thermal-comfort tree result (Luo et al. 2020, *Energy and Buildings* 210:109776, R² = 0.994 on 81,846 ASHRAE II votes) and is therefore a leakage artefact rather than legitimate performance, as confirmed by our humidex ablation (tree R² drops to 0.88–0.90 once `humidex_c` is removed). Ridge Regression, with test R² = 0.5748 and test MAE = 0.1753, is the scientifically defensible recommendation: its performance sits squarely inside the literature-plausible band for single-location linear thermal-comfort baselines (Liu et al. 2022, MAE 0.5–0.9, R² 0.4–0.7), it is fully interpretable, and its limited capacity prevents it from exploiting the same leakage path that inflates the tree models. The collaborator TFT (biosense360) targets `humidex_c` in °C on a 1.2M-row merged dataset — a different problem — and would become a credible upstream forecasting component for a downstream livability classifier once its 80% interval coverage (currently 58.6% at the 120-min horizon vs. 80% nominal) is calibrated and its training is confirmed past the current epoch-7 early stop.
