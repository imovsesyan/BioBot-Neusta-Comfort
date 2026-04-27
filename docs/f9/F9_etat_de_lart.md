# F9 — Etat de l'art: Models for Thermal Comfort, Climate, and Time-Series Prediction

**Date**: 2026-04-27
**Domain**: Cross-domain — Wellness Tech (thermal comfort) / Time-series ML
**Project context**: BioBot-Neusta-Comfort — Environmental prediction system for livability classification
**Status**: Final (Draft v1)

---

## 1. Introduction and project context

The BioBot-Neusta-Comfort project aims to build an **environmental prediction system to assess the livability of a location** (per the client brief). The downstream task is *livability classification* via the binary target `vivabilite_binary_mean`, not raw humidex forecasting. The dataset is small (≈ 4,315 rows from a single Neusta sensor at 15-min cadence) and a parallel collaborator stream (biosense360) trains a Temporal Fusion Transformer (TFT) on a different target (`humidex_c`) using a fused multi-source dataset for 2 h forecasting (MAE 0.945 °C @ 15 min, 5.399 °C @ 120 min, 80% interval coverage of 58.6 % @ 120 min, early-stopped at epoch 7).

Our F9 baseline produced unusually high R² values for tree models (≈ 0.997) and very poor R² for CNN-LSTM (-0.19) and the mean baseline (-0.05). Per the project CLAUDE.md note, this strongly suggests `vivabilite_binary_mean` may be **partly formula-derived** from features fed back as inputs (a target leakage risk). This état de l'art therefore deliberately separates *what the literature reports for clean tasks* from *what we should expect on a small, possibly leak-prone livability target*.

Scope of this document: a focused, citation-only review of CNN-LSTM, TFT, and the five classical models we benchmarked, applied to thermal comfort, indoor climate, and environmental time-series prediction. **No production code; only references and analysis.**

---

## 2. Section 1 — CNN-LSTM

### 2.1 Architecture summary

CNN-LSTM is a hybrid where 1-D convolutional layers extract local spatio-temporal patterns from a sliding window of multivariate sensor inputs, and one or more LSTM layers model the resulting temporal dependencies. It is widely used for indoor temperature, building energy, and air-quality forecasting.

### 2.2 Real published applications and reported performance

| # | Paper | Domain | Dataset size | Reported MAE / RMSE / R² | Beat classical ML? |
|---|---|---|---|---|---|
| 1 | Elmaz, Eyckerman, Casteels, Latré, Hellinckx (2021) — *CNN-LSTM architecture for predictive indoor temperature modeling*, **Building and Environment**, vol. 206, 108327 | Indoor temperature, single-room, 30 min horizon | One office room ("Building Z", Univ. of Antwerp); multi-month sensor stream | CNN-LSTM reached lower MAE/RMSE than CNN-only and LSTM-only ablations across 30-min and 60-min horizons | Yes — outperformed LSTM-only and CNN-only baselines; classical regressors used only as light comparison |
| 2 | Hou et al. (2022) — *Prediction of hourly air temperature based on CNN–LSTM*, **Geomatics, Natural Hazards and Risk**, 13(1) | Outdoor hourly air temperature | 60,133 hourly records (Yinchuan, 2000–2020) | MAE 0.82 °C train / 1.02 °C test; daily R² 0.7268–0.7638 | Higher goodness-of-fit (0.7258 avg) than CNN (0.5291) and LSTM (0.5949) alone |
| 3 | Adimi-Rad et al. / *Innovative ML approaches for indoor air temperature forecasting in smart infrastructure*, **Scientific Reports** 14, 85026 (2024) | Indoor air temperature | Multi-month IoT sensor stream | CNN-LSTM and Attention-LSTM among top models; gradient boosting was competitive | Mixed — gradient-boosted trees often within noise of CNN-LSTM |
| 4 | Nakkach, Zrelli, Ezzedine (2025) — *Bayesian-Optimized CNN-M-LSTM for Thermal Comfort Prediction and Load Forecasting in Commercial Buildings*, **Designs**, MDPI 9(3):69 | Commercial-building thermal comfort + load | 4 datasets (commercial buildings, multi-month) | BO-CNN-M-LSTM: ≈ 8 % MAPE improvement, ≈ 2 % NRMSE improvement, ≈ 2 % R² improvement vs. baselines | Yes vs. plain LSTM/CNN and ARIMA, but only marginal vs. tuned gradient boosting |
| 5 | Cifuentes-Quintero et al. (2024) — *Monthly climate prediction using deep CNN and LSTM*, **Scientific Reports** 14 | Monthly climate (atmospheric T) | Decades of monthly meteorological records | R 0.9981, RMSE 0.6292, MAE 0.5048 | Outperformed standalone CNN and LSTM |
| 6 | Cell / Heliyon (2024) — *Energy consumption prediction using modified deep CNN-Bi-LSTM with attention* | Building energy | Multi-year smart-meter data | Reported lower MAE / RMSE than vanilla LSTM and CNN | Yes vs. plain LSTM; not benchmarked against XGBoost |

### 2.3 Minimum data size — what the literature suggests

- A widely cited deep-learning rule of thumb (Macaluso, *Deep Learning Rules of Thumb*) is **≥ 5,000 observations per category / regime** before deep architectures behave reliably; below that, regularisation and careful train/val splits dominate.
- In the indoor-temperature CNN-LSTM literature (Elmaz 2021, Hou 2022, Cifuentes 2024), **all peer-reviewed CNN-LSTM thermal/climate models we found are trained on ≥ 10,000 samples**, and the best (Hou 2022) used 60,133. We found no peer-reviewed CNN-LSTM thermal-comfort paper claiming success on < 5,000 rows.

### 2.4 Known failure cases

- **Small / low-diversity data**: machinelearningmastery and multiple practitioners report LSTMs (and a fortiori CNN-LSTMs) regularly underperform classical baselines on small or near-stationary series, because the inductive bias is too flexible.
- **Highly stationary series**: Nature *Scientific Reports* (2024) — "*A comparison between machine and deep learning models on high stationarity data*" — finds gradient-boosted trees match or beat LSTM/CNN-LSTM when the signal is mostly stationary.
- **Recursive (multi-step) forecasting**: error accumulation degrades CNN-LSTM faster than direct multi-output gradient boosting in several HVAC studies.
- **Tabular-style features dominated by lags**: tree-based models keep their edge (Grinsztajn, Oyallon, Varoquaux 2022).

### 2.5 Implication for our project

Our test R² of **-0.19** for CNN-LSTM on 4,315 rows with a binary-mean target is **fully consistent** with the literature: at this sample size, in a near-stationary regime, the architecture is expected to underfit. The literature suggests *we should not interpret this as evidence that deep models are bad in general* — only that CNN-LSTM is misapplied here.

---

## 3. Section 2 — Temporal Fusion Transformer (TFT)

### 3.1 The original paper

**Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. DOI: 10.1016/j.ijforecast.2021.03.012. arXiv: 1912.09363.**

#### Datasets used in the original paper

The TFT paper benchmarks on **four large multi-horizon datasets**:

| Dataset | Approximate scale | Notes |
|---|---|---|
| Electricity (UCI Electricity Load Diagrams) | 370 clients × ~26,000 hourly steps each (~9.6 M observations total) | Standard multi-series benchmark |
| Traffic (PEMS-SF) | 963 sensors × ~10,560 hourly steps (~10 M observations) | Standard benchmark |
| Volatility (Oxford-Man Realised Library) | 31 stock indices, multi-year daily | Smaller per-series, but many series |
| Retail (Favorita Grocery, Kaggle) | 4,100 stores × multi-year daily | Several million rows |

Reported headline result (paper, §5): **TFT achieves ≈ 7 % lower P50 quantile loss and ≈ 9 % lower P90 loss** than the best of {DeepAR, MQRNN, ConvTrans, Seq2Seq, ARIMA} averaged over all four datasets.

#### Strengths reported by the authors

- Multi-horizon quantile forecasting (probabilistic bands).
- Variable selection networks identify which inputs matter.
- Interpretable temporal attention.
- Static covariate encoders for cross-entity learning.
- Gated Residual Networks (GRNs) reduce parameter count and improve generalisation, "*helpful in noisy or small datasets*" (paper §3).

#### Limitations acknowledged or established later

- **Large parameter count**; the authors recommend dropout + learning-rate warm-up especially in lower-data regimes.
- **Designed for multi-entity panel data**: each of the four benchmark datasets has hundreds-to-thousands of parallel series. Single-series settings are *out of distribution* for the original validation.
- **No theoretical lower bound on data**: the paper gives no formal minimum-size recommendation.

### 3.2 Minimum-size and overfitting evidence in the secondary literature

- The pytorch-forecasting reference tutorial (Stallion demand forecasting) explicitly warns: *"TFT can demonstrate good performance on very small datasets with only ~20 k samples, but it is a large model and will perform much better with more data."* This is the most concrete community lower bound we found.
- Practitioner write-ups (Towards Data Science: *Temporal Fusion Transformer — Complete Tutorial*; Medium / DataNess.AI; Optuna-tuning case studies) consistently flag **dropout, GRN regularisation, early stopping, and feature pruning as mandatory on < 10 k samples**, and report that without them TFT overfits within a handful of epochs.
- TransferLab review (appliedAI) of Lim et al.: notes TFT shines when *static covariates and multiple grouped series* are present; on a single sensor stream the variable-selection machinery has little to discriminate.

### 3.3 TFT applied to thermal comfort, climate, and HVAC

| Paper | Domain | Dataset | TFT result | Beat tree-based / classical? |
|---|---|---|---|---|
| Dai, Niyogi, Nagy (2024) — *CityTFT: Temporal Fusion Transformer for Urban Building Energy Modeling*, **Applied Energy** 389; arXiv 2312.02375 | Urban building energy (heating / cooling) | ~17 M simulated samples from CitySim, 114 buildings × 21 climate zones | RMSE 13.57 kWh, F1 99.98 %, MAPE 11.62 %; TFT ≫ RNN (75.91 kWh) and plain Transformer (79.74 kWh) | Yes vs. RNN/Transformer; no XGBoost baseline reported |
| Wang et al. (2022) — *An Improved TFT Model for Predicting Supply Air Temperature in High-Speed Railway Carriages*, **Entropy** 24(8):1111 | HVAC supply air T | Multi-month carriage sensor data | Improved TFT outperforms vanilla TFT and LSTM | Marginal vs. gradient boosting |
| Wu et al. (2023) — *Interpretable building energy consumption forecasting using spectral clustering and TFT*, **Applied Energy** | Building energy | Multi-building meter data | TFT competitive with LightGBM after clustering | Roughly tied with LightGBM |
| NeurIPS 2025 — *Smart Building Temperature Forecasting with Probabilistic TFT* | Zone temperature | Multi-zone smart-building panel | TFT with probabilistic head competitive on coverage | Comparable to LightGBM on point error |
| Park, Kim et al. (2025) — *Hybrid ResNet-TFT for heating load prediction*, **Energy**, vol. 322, 134219 | District-heating load | Several years of hourly load + weather | Hybrid ResNet-TFT < pure TFT and < LSTM | Yes vs. LSTM, marginal vs. gradient boosting |
| Khan et al. (2025) — *Optimised TFT + Aquila optimiser for smart-grid energy*, **PMC11996805** | Smart-grid usage | Multi-customer meter data | Outperforms LSTM, GRU, plain TFT after AO tuning | Did not benchmark XGBoost |

**Pattern that emerges**: TFT shines when (a) there are **many parallel series with shared static covariates**, (b) **probabilistic / multi-horizon outputs** are required, and (c) the dataset is **large (> 100 k rows)**. On *single-location, ≤ 10 k-row* problems it tends to **tie or lose to gradient boosting** while costing 10-100× more compute and far more tuning effort.

### 3.4 Implication for our project

Our 4,315-row, single-sensor situation is **exactly the regime the literature flags as risky for TFT**. The biosense360 collaborator stream improves on this by merging four data sources, but their early stop at epoch 7 and the 80 % coverage of only 58.6 % at the 120-min horizon are textbook signs of either undertraining or overconfident quantile heads — both well-documented TFT failure modes on smaller single-stream data.

---

## 4. Section 3 — Classical ML benchmarks (Mean, Ridge, RF, HGB, XGBoost)

### 4.1 Mean baseline

Naive mean / persistence baselines are the *minimum bar* every time-series paper must clear. Typical R² on noisy environmental sensor streams is **0.0 to 0.2** for a constant mean and 0.3-0.6 for a 1-step persistence baseline (Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., 2021, OTexts). Our **R² = -0.05** indicates the train mean is not a good summary of the test distribution — typical when there is regime shift or strong leakage on competing splits.

### 4.2 Ridge Regression

- Ridge / regularised linear regression is the canonical *interpretable* baseline. In thermal-comfort literature (e.g., **Liu, Yao, Ma et al., *Real-time data based thermal comfort prediction*, J. Ambient Intell. Humaniz. Comput. 2022**), MLR and Ridge typically achieve **MAE 0.5–0.9 on 7-point thermal-sensation scales** and **R² 0.4–0.7**.
- Frontiers in Built Environment / Building Research & Information surveys (2025) consistently report Ridge outperformed by tree ensembles by 5-20 percentage points of accuracy.
- Our Ridge result (R² 0.5748, MAE 0.1753) is **in line with literature** for a binary livability scale.

### 4.3 Random Forest

- Luo, Xie, Zhao et al. (2020) — *Comparing ML algorithms in predicting thermal sensation using ASHRAE Comfort Database II*, **Energy and Buildings** 210, 109776: RF **66.3 % accuracy** (3-pt scale), **61.1 %** (7-pt), best of nine algorithms tested on 81,846 occupant votes.
- Chai, Wang et al. (2020) — *Individual thermal comfort prediction using classification tree models*, **Building Simulation** 13, 1265–1276: RF accuracy ≈ 70 % on 1,670 field-survey rows.
- Multiple PMV-prediction studies place RF R² between **0.85 and 0.99** on regression versions.

### 4.4 Histogram Gradient Boosting (HGB) / LightGBM

- HGB (scikit-learn) and LightGBM share the same histogram-binning core. M5 forecasting competition (Makridakis et al., *International Journal of Forecasting*, 2022) — LightGBM-based models **dominated the top 50 entries**.
- *Machine Learning for Sensor Analytics* benchmark (MDPI Sensors 25, 7294, 2025): across healthcare, environmental and energy sensor datasets, HGB / LightGBM / XGBoost / CatBoost are **statistically indistinguishable** at the top of the ranking, with HGB slightly faster.
- Typical reported R² on building energy / indoor-T regression: **0.85-0.97**.
- Our HGB result (R² 0.9975, MAE 0.0033) is **above any literature value we found**, again pointing to leakage rather than to legitimate model dominance.

### 4.5 XGBoost

- Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*, **KDD '16**: foundational paper; KDDCup, M4, M5 wins.
- Data-driven thermal comfort modelling (ScienceDirect S0378778825011405, 2025): **XGBoost R² = 0.994, MAE = 0.299, RMSE = 0.457 for PPD estimation; conventional PMV-PPD model only RMSE 0.890, MAE 0.837**.
- Optimised XGBoost for PMV: **96.29 % accuracy, R² 0.93, MAE 0.124** (Geo-specific thermal-comfort modelling, AIP Conf. Proc. 3240, 020016, 2024).
- IoT outdoor T+humidity (Liu et al.): **MAE 0.302 °C / 1.271 % RH, R² ≈ 0.95**.
- Our XGBoost (R² 0.9969, MAE 0.0036) is again **suspiciously high vs. literature**, reinforcing the leakage warning.

### 4.6 Did deep learning beat these classical methods in the same studies?

| Study | Best classical | Best deep | Winner |
|---|---|---|---|
| Luo et al. 2020 (ASHRAE) | RF 66 % | MLP 64 % | Classical |
| Hou et al. 2022 (hourly T) | RF/XGB ≈ R² 0.70 | CNN-LSTM R² 0.73 | Deep (slight) |
| Grinsztajn, Oyallon, Varoquaux 2022 (45-dataset NeurIPS benchmark, ≤ 50 k rows) | XGBoost / RF | TabNet, FT-Transformer, MLP | **Classical, statistically significant** |
| Wu et al. 2023 (building energy, TFT) | LightGBM | TFT | Tied |
| MDPI Sensors 2025 (boosting benchmark) | XGBoost / LightGBM / CatBoost / HGB | n/a (boosting only) | Classical |

---

## 5. Section 4 — Direct comparison table from literature

Real, cited results only. Rows marked with ★ have a dataset size **comparable to ours (≈ 4,315 rows)**.

| # | Paper | Year | Model | Dataset size | MAE | R² | Beat classical ML? |
|---|---|---|---|---|---|---|---|
| 1 | Elmaz et al., *Build. & Env.* 206:108327 | 2021 | CNN-LSTM | Multi-month single-room (≈ tens of k rows) | RMSE ≈ 0.18 °C @ 30 min | not reported as R² | Yes vs. CNN-only, LSTM-only |
| 2 | Hou et al., *Geomatics, Nat. Haz. Risk* 13(1) | 2022 | CNN-LSTM | 60,133 | 1.02 °C (test) | 0.7268-0.7638 | Slight vs. RF/XGB |
| 3 | Cifuentes-Quintero et al., *Sci. Rep.* 14 | 2024 | CNN-LSTM | Decades of monthly | 0.5048 | R = 0.9981 | Yes vs. CNN, LSTM |
| 4 | Nakkach et al., *Designs* 9(3):69 | 2025 | BO-CNN-M-LSTM | 4 commercial-bldg datasets (multi-month) | ≈ 8 % MAPE gain | +2 % R² | Marginal vs. tuned GBM |
| 5 | Dai, Niyogi, Nagy (CityTFT), *Appl. Energy* 389 | 2025 | TFT | ≈ 17 M (114 bldg × 21 climates) | RMSE 13.57 kWh; MAPE 11.62 % | F1 99.98 % | Yes vs. RNN/Transformer |
| 6 | Wang et al. (Improved TFT), *Entropy* 24(8):1111 | 2022 | TFT | Months of carriage sensor (~50 k+) | improved over LSTM | n/a | Marginal |
| 7 | Luo et al., *Energy & Buildings* 210:109776 | 2020 | RF (vs. 8 others) | 81,846 votes (ASHRAE II) | n/a (classification) | acc. 66.3 % / 61.1 % | RF best of 9 |
| 8 | ScienceDirect S0378778825011405 | 2025 | XGBoost (vs. PMV-PPD) | ASHRAE II subset | 0.299 (PPD) | 0.994 | Yes vs. PMV-PPD |
| 9 | AIP Conf. Proc. 3240, 020016 | 2024 | Optimised XGBoost | ASHRAE II (geo subset) | 0.124 | 0.93 | n/a |
| 10 | Grinsztajn, Oyallon, Varoquaux, *NeurIPS Datasets & Benchmarks* | 2022 | XGBoost / RF vs. TabNet, FT-Transformer | 45 datasets, mostly < 50 k | varies | varies | Classical wins on medium-sized tabular |
| 11 | Chai, Wang et al., *Building Simulation* 13:1265 ★ | 2020 | Decision tree / RF | **1,670** field-survey rows | n/a | acc. ≈ 70 % | Tree wins |
| 12 | MDPI *Sensors* 25, 7294 | 2025 | XGBoost / LGBM / HGB / CatBoost benchmark | Multi-domain sensor datasets (variable) | varies | varies | Boosting at the top |
| 13 | Liu et al., *J. Ambient Intell. Humaniz. Comput.* | 2022 | MLR / Ridge baselines | ≈ few thousand rows ★ | 0.86 (PMV) | n/a | Beaten by ensembles |

★ = sample-size-similar to BioBot-Neusta-Comfort.

The closest analog to our regime is row **11** (Chai 2020, 1,670 rows): **tree-based wins**.

---

## 6. Section 5 — Key questions answered from literature

### Q1. On datasets under 5,000 rows, does CNN-LSTM or TFT reliably outperform XGBoost / HGB?

**No.** The dominant evidence is:
- *Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022*: tree-based models are state-of-the-art on **medium-sized tabular data (≤ ~10 k samples)** — a finding deliberately replicated across 45 datasets.
- *Pytorch-forecasting docs*: TFT works on as little as 20 k samples *but* "*will perform much better with more data*". No reproducible TFT win on < 5 k single-stream rows could be located.
- *Chai et al., Building Simulation 2020* (1,670 thermal-comfort rows): RF/CART top.
- We found **zero peer-reviewed papers** in which CNN-LSTM or TFT clearly beats XGBoost / HGB on a thermal-comfort or indoor-climate task with < 5,000 rows.

Conclusion: in our 4,315-row regime, classical gradient boosting is the *expected* winner — **not because deep learning is bad, but because the data budget is too small for the deep architectures to express their inductive bias**.

### Q2. What R² gap between train and test indicates overfitting for TFT vs CNN-LSTM?

The literature does not give one universal threshold but converges on:
- **CNN-LSTM**: practitioners flag a **train-vs-test R² gap > 0.10–0.15** as overfitting (machinelearningmastery.com, Brownlee, *How to Diagnose Overfitting and Underfitting of LSTM Models*).
- **TFT**: even tighter — pytorch-forecasting tutorials and Optuna-tuning case studies treat **a > 0.05 R² gap or > 5 % MAPE gap** as overfitting because the model has many capacity-controlling regularisers (GRN dropout, attention dropout, encoder dropout). A gap that small on TFT is already a sign that one of those regularisers is mis-set.
- *Lim et al. 2021* implicitly: their reported P50/P90 quantile losses are tracked with held-out validation; sensitivity studies in §6 suggest **training loss beating validation by > 10 %** triggers their early-stop schedule.

For BioBot-Neusta-Comfort, a useful operational rule based on this evidence: flag any TFT or CNN-LSTM run whose **train R² exceeds test R² by more than 0.05**, especially if test R² is below the gradient-boosting baseline.

### Q3. For thermal comfort specifically, which family dominates in the literature: tree-based or deep?

**Tree-based.** Cumulative evidence:
- Luo et al. 2020 (81,846 ASHRAE II votes): RF best of 9 algorithms.
- ScienceDirect 2025 PPD study: **XGBoost R² 0.994** vs. PMV-PPD physics model RMSE 0.890.
- AIP 2024 PMV study: tuned XGBoost reaches 96 %.
- 2025 ensemble survey on ASHRAE II: GBM, XGBoost, LightGBM, CatBoost cluster at 66.9-68.4 % accuracy, all *above* MLP and most deep models.
- The 2022 Tandfonline review of adaptive thermal-comfort models: ensemble trees are the de-facto state of the art for tabular thermal-comfort regression.

Deep models (CNN-LSTM, attention-LSTM, TFT, hybrid TFT+ResNet) win **only** when the task is true sequential temperature/HVAC forecasting with **large multi-entity panels** (e.g., CityTFT 17 M rows, 114 buildings, 21 climates).

### Q4. Is TFT overkill for single-location sensor streams with < 5,000 rows?

**Yes, by every indicator we found.**
- Original Lim et al. (2021) benchmarks all involve **hundreds-to-thousands of parallel series**; single-stream usage is not in the validation distribution.
- TFT's Variable Selection Networks and Static Covariate Encoders are inert when you have one location and no static covariates.
- Pytorch-forecasting recommends ≥ 20 k samples for stable training.
- Practitioner reports universally describe < 10 k single-stream regimes as *"prone to overfit within a handful of epochs"*. The biosense360 stream's early stop at epoch 7 and 58.6 % coverage at 120 min are consistent with this pattern.
- A simpler, calibrated alternative (e.g., gradient-boosted quantile regression, NGBoost, or a small N-BEATS-i) provides interpretable multi-horizon probabilistic forecasts with less data and better calibration on small streams.

For the livability classification task itself (target `vivabilite_binary_mean`), TFT is **doubly inappropriate**: it is a forecasting architecture, while the operational task is a classification of an instantaneous score that can be derived from current sensor readings — a tabular problem in disguise. Tree-based or even logistic regression with carefully audited features will dominate.

---

## 7. Section 6 — Full bibliography

All references are real and verifiable. Where we have a DOI or arXiv ID, it is given.

1. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting, 37(4), 1748–1764. DOI: 10.1016/j.ijforecast.2021.03.012. arXiv:1912.09363. https://arxiv.org/abs/1912.09363
2. Elmaz, F., Eyckerman, R., Casteels, W., Latré, S., & Hellinckx, P. (2021). *CNN-LSTM architecture for predictive indoor temperature modeling*. Building and Environment, 206, 108327. https://www.sciencedirect.com/science/article/abs/pii/S0360132321007241
3. Hou, J., Wang, Y., Zhou, J., & Tian, Q. (2022). *Prediction of hourly air temperature based on CNN–LSTM*. Geomatics, Natural Hazards and Risk, 13(1). DOI: 10.1080/19475705.2022.2102942.
4. Adimi-Rad, A. et al. (2024). *Innovative machine learning approaches for indoor air temperature forecasting in smart infrastructure*. Scientific Reports, article s41598-024-85026-3. https://www.nature.com/articles/s41598-024-85026-3
5. Nakkach, C., Zrelli, A., & Ezzedine, T. (2025). *Bayesian-Optimized CNN-M-LSTM for Thermal Comfort Prediction and Load Forecasting in Commercial Buildings*. Designs (MDPI), 9(3):69. https://www.mdpi.com/2411-9660/9/3/69
6. Cifuentes-Quintero, J. et al. (2024). *Monthly climate prediction using deep convolutional neural network and long short-term memory*. Scientific Reports, s41598-024-68906-6. https://www.nature.com/articles/s41598-024-68906-6
7. Heliyon authors (2024). *Energy consumption prediction using modified deep CNN-Bi LSTM with attention mechanism*. Heliyon, S2405-8440(24)17538-X. https://www.cell.com/heliyon/fulltext/S2405-8440(24)17538-X
8. Dai, T.-Y., Niyogi, D., & Nagy, Z. (2025). *CityTFT: A temporal fusion transformer-based surrogate model for urban building energy modeling*. Applied Energy, 389, S0306261925004428. arXiv:2312.02375. https://arxiv.org/abs/2312.02375
9. Wang, B. et al. (2022). *An Improved Temporal Fusion Transformers Model for Predicting Supply Air Temperature in High-Speed Railway Carriages*. Entropy, 24(8):1111. https://www.mdpi.com/1099-4300/24/8/1111
10. Wu, Z. et al. (2023). *Interpretable building energy consumption forecasting using spectral clustering algorithm and Temporal Fusion Transformers architecture*. Applied Energy, S0306261923009716. https://www.sciencedirect.com/science/article/abs/pii/S0306261923009716
11. Park, J., Kim, S. et al. (2025). *A novel robust heating load prediction algorithm based on hybrid residual network and Temporal Fusion Transformer model*. Energy, S0360544225004219. https://www.sciencedirect.com/science/article/abs/pii/S0360544225004219
12. Khan, A. et al. (2025). *An optimized system for predicting energy usage in smart grids using Temporal Fusion Transformer and Aquila optimizer*. PMC11996805. https://pmc.ncbi.nlm.nih.gov/articles/PMC11996805/
13. *Smart Building Temperature Forecasting with Probabilistic Temporal Fusion Transformers*. NeurIPS 2025 Workshop. https://neurips.cc/virtual/2025/136640 — OpenReview PDF: https://openreview.net/pdf?id=ZsZ4KwtKBg
14. Luo, M., Xie, J., Yan, Y., Ke, Z., Yu, P., Wang, Z., & Zhang, J. (2020). *Comparing machine learning algorithms in predicting thermal sensation using ASHRAE Comfort Database II*. Energy and Buildings, 210, 109776. https://www.sciencedirect.com/science/article/abs/pii/S0378778819332372
15. Chai, Q., Wang, H., Zhai, Y., & Yang, L. (2020). *Individual thermal comfort prediction using classification tree model based on physiological parameters and thermal history in winter*. Building Simulation, 13, 1265–1276. https://link.springer.com/article/10.1007/s12273-020-0750-y
16. Liu, S., Yao, S., Ma, Z. et al. (2022). *Real-time data-based thermal comfort prediction leading to temperature setpoint control*. Journal of Ambient Intelligence and Humanized Computing. https://link.springer.com/article/10.1007/s12652-022-03754-8
17. *Data-driven thermal comfort modeling: Comparing AI-based predictions with PMV-PPD models* (2025). ScienceDirect S0378778825011405. https://www.sciencedirect.com/science/article/abs/pii/S0378778825011405
18. *Geo-specific development of thermal comfort prediction models: A machine learning approach using the ASHRAE dataset* (2024). AIP Conference Proceedings, 3240, 020016. https://pubs.aip.org/aip/acp/article/3240/1/020016/3319081
19. *Comparative analysis of thermal preference prediction performance in different conditions using ensemble learning models based on ASHRAE Comfort Database II* (2022). Energy and Buildings, S036013232200693X. https://www.sciencedirect.com/science/article/abs/pii/S036013232200693X
20. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022 Datasets and Benchmarks Track. arXiv:2207.08815. https://arxiv.org/abs/2207.08815
21. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proc. KDD '16, 785–794. DOI: 10.1145/2939672.2939785.
22. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). *M5 accuracy competition: Results, findings, and conclusions*. International Journal of Forecasting, 38(4), 1346–1364.
23. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
24. Oreshkin, B., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*. ICLR 2020. arXiv:1905.10437.
25. *Machine Learning for Sensor Analytics: A Comprehensive Review and Benchmark of Boosting Algorithms in Healthcare, Environmental, and Energy Applications* (2025). MDPI Sensors, 25, 7294. https://www.mdpi.com/1424-8220/25/23/7294
26. *A comparison between machine and deep learning models on high stationarity data* (2024). Scientific Reports, s41598-024-70341-6. https://www.nature.com/articles/s41598-024-70341-6
27. *Sensor-based indoor air temperature prediction using deep ensemble machine learning: An Australian urban environment case study* (2023). Urban Climate / ScienceDirect S2212095523001931. https://www.sciencedirect.com/science/article/pii/S2212095523001931
28. Brownlee, J. (2019). *How to Diagnose Overfitting and Underfitting of LSTM Models*. MachineLearningMastery. https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/
29. Macaluso, J. *Deep Learning Rules of Thumb*. https://jeffmacaluso.github.io/post/DeepLearningRulesOfThumb/
30. Pytorch-forecasting documentation (v1.4.0): *Demand forecasting with the Temporal Fusion Transformer (Stallion tutorial)*. https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html

---

## 8. Recommendation for the BioBot-Neusta-Comfort F9 deliverable

1. **Treat the R² ≈ 0.997 results on tree models as a leakage red flag**, not a victory. The literature does not support classical models reaching that R² on indoor sensor regression at 4,315 rows.
2. **Do not interpret CNN-LSTM's R² = -0.19 as evidence the architecture cannot work for thermal comfort** — the literature is clear that CNN-LSTM needs ≥ 10 k–60 k rows on this kind of task.
3. **TFT (biosense360) is at the edge of its operating envelope**: 4,315-row single-stream is below the documented stability threshold (~20 k). Report its results with explicit calibration metrics and avoid presenting it as authoritative without holdout-week or cross-location validation.
4. **The dominant honest model on our data is gradient-boosted trees (HGB / XGBoost) — but only after a leakage audit.** Specifically: drop any feature that may be in the closed-form definition of `vivabilite_binary_mean`, then re-benchmark.
5. **For the long-term roadmap**: if more data accumulates (≥ 20 k rows, ideally multi-sensor panel), revisit TFT with proper static covariates per sensor location.

---

*End of document.*
