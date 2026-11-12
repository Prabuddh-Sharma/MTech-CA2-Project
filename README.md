# Predictive Maintenance: Machine Failure Prediction on the AI4I 2020 Dataset

**Course:** M.Tech Machine Learning — Project-Based Continuous Assessment (CA2)  
**Author:** Prabuddh Sharma  
**GitHub Repository:** [https://github.com/Prabuddh-Sharma/MTech-CA2-Project](https://github.com/Prabuddh-Sharma/MTech-CA2-Project)

---

## 1. Project Overview
Unplanned milling machine breakdowns cause costly production halts and safety risks. This project builds an end-to-end, leakage-free Machine Learning classification pipeline on the **UCI AI4I 2020 Predictive Maintenance Dataset** to predict `Machine failure` (`1` = Failure, `0` = Normal) from operational sensor telemetry, deployed via an interactive **Streamlit** dashboard for plant maintenance engineers.

## 2. Dataset & Citation
* **Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset (Dataset ID: `601`, DOI: `10.24432/C5HS5C`, License: `CC BY 4.0`).
* **Primary Reference:** S. Matzka, *"Explainable Artificial Intelligence for Predictive Maintenance Applications,"* 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), pp. 69–74, DOI: `10.1109/AI4I49448.2020.00023`.
* **Dataset Properties:** 10,000 instances, 14 columns, 0 missing values, 0 duplicate rows, and an imbalanced failure rate of **3.39%** (339 failures vs. 9,661 normal runs). *Note: The dataset is synthetically generated from physical milling rules.*

## 3. Methodology Highlights
1. **Leakage Prevention:** Removed non-predictive identifiers (`UDI`, `Product ID`) and post-event failure sub-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) that define `Machine failure`.
2. **Physics-Guided Feature Engineering (`features.py`):**
   * `temp_diff` $[\text{K}] = \text{Process temperature [K]} - \text{Air temperature [K]}$
   * `power` $[\text{W}] = \text{Torque [Nm]} \times \left(\text{Rotational speed [rpm]} \times \frac{2\pi}{60}\right)$
3. **Statistical Validation:** Confirmed significant dependency between `Type` and failure ($\chi^2 = 13.7517, p = 0.00103$) and estimated the 95% Wilson Score Confidence Interval for the population failure rate at **$[3.05\%, 3.76\%]$**.
4. **Leakage-Free Modeling:** Stratified 80/20 train-test split (`8,000` train / `2,000` test, `random_state=42`), Stratified 5-Fold Cross-Validation, `StandardScaler` applied strictly inside pipelines for scale-sensitive models (Logistic Regression, KNN), and `class_weight='balanced'` / `'balanced_subsample'` for class imbalance.

## 4. Model Comparison Summary (Held-Out Test Set: 2,000 Machines, 68 Failures)

| Model | CV F1 (Mean ± Std) | Test F1 | Test Recall | Test Precision | Test ROC-AUC | Test PR-AUC | Test Accuracy | TP / FN / FP / TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0340 | 0.9660 | 0 / 68 / 0 / 1932 |
| **Logistic Regression (Tuned)** | 0.2744 ± 0.0181 | 0.2943 | **0.8676** | 0.1772 | 0.9340 | 0.4659 | 0.8585 | 59 / 9 / 274 / 1658 |
| **KNN (Tuned, k=3, distance)** | 0.5148 ± 0.0463 | 0.5565 | 0.4706 | 0.6809 | 0.8211 | 0.5446 | 0.9745 | 32 / 36 / 15 / 1917 |
| **Decision Tree (Tuned)** | 0.7356 ± 0.0525 | 0.7344 | 0.6912 | 0.7833 | 0.8422 | 0.5519 | 0.9830 | 47 / 21 / 13 / 1919 |
| **Random Forest (Tuned — Selected)** | **0.8207 ± 0.0328** | **0.8594** | 0.8088 | **0.9167** | **0.9689** | **0.8598** | **0.9910** | **55 / 13 / 5 / 1927** |

## 5. Repository Structure
```text
MTech-CA2-Project/
├── data/ai4i2020.csv                 # Raw UCI AI4I 2020 dataset
├── notebooks/predictive_maintenance.ipynb # Complete EDA, statistics, and modeling notebook
├── models/final_model.joblib           # Serialized best Random Forest pipeline
├── outputs/figures/                    # Saved high-resolution EDA and evaluation plots
├── report/                             # Final project report
├── features.py                         # Shared feature engineering module
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation