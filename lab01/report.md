# Lab 1: Environment and First System Measurements

## 1. Goal
To establish a reproducible and isolated Python 3.11.8 environment, train baseline linear (`LogisticRegression`) and ensemble (`RandomForestClassifier`) models on the Breast Cancer Wisconsin dataset using a stratified 70/30 split, and evaluate their deployment suitability against hardware resource budgets.

## 2. Method
* **Environment**: Python 3.11.8 with pinned dependencies tracked via `requirements.txt` and `versions.txt`.
* **Dataset**: Breast Cancer Wisconsin (Diagnostic) loaded via `scikit-learn`, split 70/30 with stratification and `random_state=42`.
* **Reproducibility**: Fixed global random seeds (`42`) across `random`, `numpy`, and `torch`.
* **Measurements**: Training time measured via median of 5 runs after 1 warm-up; single-sample inference latency measured over 100 repetitions; peak RSS memory captured using `psutil`; model size serialized with `joblib`.

## 3. Results Table

| Model | Accuracy | Median Train Time (s) | Median Inference Latency (ms) | Model Size (KB) | Peak RSS Memory (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LogisticRegression | 0.9415 | [Insert value] | [Insert value] | 1.03 | 268.30 |
| RandomForestClassifier | 0.9357 | [Insert value] | [Insert value] | 284.07 | 268.93 |

## 4. Conclusions
1. **Performance Parity**: Both Logistic Regression and Random Forest achieved high test accuracies above 93.5%, with Logistic Regression slightly outperforming the Random Forest model on this dataset.
2. **Deployment Fit**: Due to peak memory usage around 268 MB, both models qualify for deployment under the **Edge** resource budget tier, while their compact model sizes (under 300 KB) easily meet memory constraints for smaller hardware footprints.
3. **Reproducibility & Systems Engineering**: Pinning package versions and enforcing strict seed configurations ensured consistent, deterministic results across script executions, highlighting the importance of system-level tracking in ML engineering.