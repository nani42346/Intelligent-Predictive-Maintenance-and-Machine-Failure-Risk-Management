# Intelligent Predictive Maintenance & Machine Failure Risk Management

Binary classification of machine failure on the AI4I 2020 Predictive Maintenance
dataset. The project compares Logistic Regression, Random Forest and an ANN
(TensorFlow/Keras), audits their reliability, runs a controlled ANN
learning-rate experiment, and turns the predictions into risk bands and a
prioritised inspection queue.

## Dataset
- **Source:** UCI Machine Learning Repository – AI4I 2020 Predictive Maintenance Dataset
  <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- **File:** `data/ai4i2020.csv` (10,000 records, 14 columns), the unmodified original.
- **Target:** `Machine failure` (0 = no failure, 1 = failure; 3.39% positive).
- **Excluded from features:** `UDI` and `Product ID` (identifiers), and `TWF`,
  `HDF`, `PWF`, `OSF`, `RNF` (failure modes; leakage).

## How to run
1. Python 3.10 or newer is recommended. The notebook was run with Python 3.14.0.
2. `pip install -r requirements.txt`
3. Open `notebooks/Predictive_Maintenance.ipynb` (from the project root or from `notebooks/`; the
   notebook locates `data/ai4i2020.csv` and `results/` automatically).
4. Run **Kernel → Restart & Run All**. A full run takes
   about 15 minutes on a CPU, most of it the 15 ANN runs in Phase 4.
5. Outputs are written to `results/`.

## Folder structure
```
Koteswara_Rao_PredictiveMaintenance/
├── notebooks/Predictive_Maintenance.ipynb   executable notebook with explanations and outputs
├── reports/Final_Technical_Report.pdf       problem framing, methodology, results, audit, experiment, recommendations
├── results/                                 CSV outputs written by the notebook
├── data/ai4i2020.csv                        dataset (UCI AI4I 2020)
├── README.md
└── requirements.txt
```

## Outputs (`results/`)
| File | Contents |
|---|---|
| `model_comparison.csv` | Every model on Train, Validation and Test, at the default 0.5 and the validation cost-optimal threshold: all metrics, confusion counts and hypothetical cost |
| `experiment_results.csv` | One row per run (ANN learning rate × 5 seeds, plus Random Forest × 5 seeds) with every setting and seed |
| `experiment_summary.csv` | Mean, standard deviation and number of runs per experiment |
| `statistical_comparison.csv` | Welch t-tests and 95% confidence intervals for the hypothesis tests |

## Method summary
- **Split:** 70 / 15 / 15, stratified, `random_state = 42`. Preprocessing
  (StandardScaler + OneHotEncoder) and class weights are fitted on the training
  set only.
- **Engineered features:** temperature difference, power (torque × speed × 2π/60)
  and strain (tool wear × torque). These are row-wise formulas with no fitted
  parameters, so they cannot leak.
- **Model and threshold selection** use validation data only. The test set is
  used once, after selection.
- **Reproducibility:** fixed seeds (42; Phase 4 uses 42, 52, 62, 72, 82) and
  `tf.config.experimental.enable_op_determinism()`.

## Key results (test set)
Selected model: **Tuned Random Forest at threshold 0.15**. Recall 92.2% (47 of 51
failures caught), precision 38.8%, PR-AUC 0.892, hypothetical cost ₹3,48,000.
The ANN does **not** outperform the Random Forest: mean validation PR-AUC over
5 seeds was 0.726 vs 0.870 (p < 0.001).

## Assumptions
- **Costs are hypothetical:** ₹2,000 per unnecessary inspection and ₹50,000 per
  missed failure, with sensitivity checked at 10:1 and 50:1.
- **Risk-band cut-offs** (0.15 / 0.50) are tied to the selected threshold. They
  are initial choices to be agreed with maintenance experts.
- **Label noise is kept as supplied:** 9 failures have no failure mode, and 18
  RNF records are not labelled as failures.
- The dataset is synthetic and not a per-machine time series, so results must be
  re-validated on real plant data with a time-based split.
