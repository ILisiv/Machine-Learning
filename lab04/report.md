# Problem Definition: UCI Adult Income Prediction

## 1. Overview and Objective
The goal of this project is to build a robust, reproducible, and leakage-free machine learning pipeline to predict whether an individual's income exceeds $50,000 per year based on census data.

## 2. Dataset and Features
* **Dataset**: UCI Adult dataset loaded via OpenML (`fetch_openml(name='adult', version=2, as_frame=True)`)[cite: 1].
* **Inputs (14 Features)**: Mixed numerical and categorical attributes including age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country.
* **Target Output**: Binary classification label (`income` / `>50K` vs `<=50K`).

## 3. Success Metrics
* **Primary Metrics**: **F1-Score** and **Accuracy** to evaluate classification performance properly, especially given potential class imbalance.

## 4. Constraints and Requirements
* **No Data Leakage**: All transformations (imputation, scaling, encoding) must be learned exclusively from the training fold using `scikit-learn` pipelines.
* **Data Quality Checks**: Automated verification of schema, value ranges, missing values, duplicates, and label integrity.
* **Reproducibility**: Fixed random seeds and artifact hashing (SHA-256) for processed datasets and configurations.