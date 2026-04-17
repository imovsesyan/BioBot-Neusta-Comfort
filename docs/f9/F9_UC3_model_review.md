# F9-UC3 Comparative Review of Relevant Models

## Objective

Review relevant models for livability score prediction and choose a scientifically defensible testing order.

This review uses the provided research documents as project references:

- `livability_research_report.docx`
- `livability_comparison_paper.docx`
- related PDF exports provided with the project

## Model Families

| Model family | Strength | Weakness | Role in this project |
|---|---|---|---|
| Mean baseline | Shows the minimum reference performance | Not predictive | Required sanity check |
| Linear/Ridge regression | Simple, fast, stable | Cannot capture strong nonlinear rules well | First interpretable regression baseline |
| Random Forest | Strong nonlinear tabular baseline | Weak extrapolation, no native sequence memory | Current best practical baseline |
| Gradient boosting | Strong tabular model, often excellent on structured data | Needs careful validation | Main tabular benchmark |
| LSTM | Learns temporal dependencies | Needs more data, can overfit | Advanced experiment only |
| CNN-LSTM | Learns local feature interactions plus time dynamics | Higher complexity, less interpretable | Advanced candidate when target and data are mature |
| Transformer | Strong for large multimodal sequence data | Data-hungry and expensive | Future option, not justified yet |
| Ensemble or residual hybrid | Can combine tabular and sequence strengths | More complex pipeline | Future optimization path |

## Recommended Testing Order

1. Mean baseline.
2. Ridge regression.
3. Random Forest.
4. Gradient boosting.
5. LSTM or CNN-LSTM.
6. Hybrid or Transformer only if the dataset grows and the target is validated.

## Why Not Start Directly With Deep Learning?

The current Neusta target has only 4,315 usable aggregated intervals. Also, the target may be generated from environmental formulas. A deep model may overfit or appear impressive for the wrong reason.

For this reason, tree-based tabular baselines are more defensible at this stage.

## Current Model Choice

For F9-UC7, the current implemented comparison is:

- mean baseline,
- ridge regression,
- random forest,
- histogram gradient boosting,
- XGBoost.

For F9-UC8, a CNN-LSTM smoke test is implemented as an optional advanced experiment.

The current evidence favors tabular tree-based machine learning over CNN-LSTM for this dataset.

## Scientific Interpretation

If tree models perform nearly perfectly, the first conclusion should not be "the model is finished." The first conclusion should be:

> The current target may be deterministic or highly formula-derived from the input features. We need to verify the target definition before treating model performance as evidence of real comfort prediction.
