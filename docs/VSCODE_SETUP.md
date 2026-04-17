# VS Code Setup

This guide explains how to open and use the clean BioBot / Neusta repository in Visual Studio Code.

## Open the Project

From Terminal:

```bash
cd /Users/inesamovsesyan/Documents/Playground/BioBot-Neusta-Comfort
code .
```

If the `code` command is not available:

1. Open VS Code.
2. Press `Cmd + Shift + P`.
3. Search for `Shell Command: Install 'code' command in PATH`.
4. Run it.
5. Try `code .` again.

You can also open it from VS Code:

1. Open VS Code.
2. Choose `File > Open Folder`.
3. Select:

```text
/Users/inesamovsesyan/Documents/Playground/BioBot-Neusta-Comfort
```

## Create the Python Environment

In the VS Code terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Select the Interpreter

1. Press `Cmd + Shift + P`.
2. Search `Python: Select Interpreter`.
3. Select:

```text
.venv/bin/python
```

The repository includes `.vscode/settings.json` so VS Code should usually detect this automatically.

## Run the Pipeline in VS Code

Open the integrated terminal and run:

```bash
python scripts/f8_uc3_convert_to_standard_csv.py
python scripts/f8_uc4_clean_impute_normalize_aggregate.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f8_uc4_make_quality_figures.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc6_humidex_threshold_analysis.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc7_test_livability_models.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_compare_ml_dl_models.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc1_define_livable_dangerous_periods.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc3_generate_rule_alerts.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc4_train_risk_classifier.py
```

For the optional F9-UC8 CNN-LSTM experiment, install the advanced dependencies first:

```bash
python -m pip install -r requirements-advanced.txt
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8
```

You can also use the Run and Debug panel. This repository includes VS Code launch buttons for F8-UC3, F8-UC4, F9-UC6, F9-UC7, F9 ML vs DL comparison, the optional F9-UC8 CNN-LSTM experiment, and the F10 risk-detection scripts.

## XGBoost on macOS

XGBoost needs the macOS OpenMP runtime. If XGBoost fails with a `libomp.dylib` error, run:

```bash
brew install libomp
```

## Connect Codex and VS Code

You do not need to manually connect Codex to VS Code. Codex and VS Code can work on the same folder as long as both are opened at:

```text
/Users/inesamovsesyan/Documents/Playground/BioBot-Neusta-Comfort
```

Recommended workflow:

1. Open the project in VS Code.
2. Ask Codex to edit or explain files in this folder.
3. Use VS Code to inspect changes, run scripts, and view documentation.
4. Use git to commit when the changes look correct.

## Useful VS Code Extensions

Recommended extensions:

- Python
- Jupyter
- Ruff
- GitHub Pull Requests
- Markdown Preview Mermaid Support, optional
