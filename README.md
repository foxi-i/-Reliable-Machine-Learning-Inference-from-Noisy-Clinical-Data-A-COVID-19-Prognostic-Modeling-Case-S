# Reliable Machine Learning Inference from Noisy Clinical Data:A COVID-19 Prognostic Modeling Case Study


A modular preprocessing framework for improving the robustness of machine learning models trained on clinical electronic health record (EHR) data.

---

## Overview

Electronic health record (EHR) data frequently suffer from multiple data quality challenges, including noisy observations, outliers, redundant majority-class samples, and severe class imbalance. These issues often reduce model robustness and generalizability when addressed independently.

This repository provides the official implementation of **IFOSS (Isolation Forest + One-Sided Selection)**, a modular preprocessing framework that combines unsupervised outlier detection with informed undersampling before model development.

In addition to the IFOSS implementation, the repository contains benchmarking, visualization, ablation, and simulation studies used to evaluate the proposed framework.

---

## Repository Structure

```text
.
├── IFOSS.py
├── Benchmark
├── External_Benchmark
├── Simulation_ABLATION_study
├── Simulation_noise_study
├── umap_IFOSS_visualization.py
├── requirements.txt
└── README.md
```

---

## Scripts

### IFOSS.py

Standalone implementation of the proposed IFOSS preprocessing framework.

Main functionality:

* Hyperparameter optimization using Optuna
* Isolation Forest outlier detection
* One-Sided Selection undersampling
* Export of the final resampled dataset

**Input**

* `X` — feature matrix (`pandas.DataFrame`)
* `y` — binary target labels (`pandas.Series` or `NumPy array`)

**Output**

* `X_resampled`
* `y_resampled`
* Optimized IFOSS hyperparameters

---

### Benchmark

Benchmarks predictive performance before and after IFOSS preprocessing.

Implemented classifiers include:

* Logistic Regression
* Support Vector Machine (RBF)
* Random Forest
* XGBoost
* LightGBM
* CatBoost

Evaluation metrics:

* Area Under the ROC Curve (AUC)
* Weighted F1-score
* Accuracy
* Balanced Accuracy
* G-Mean

Outputs:

* Baseline performance
* Performance after IFOSS
* Absolute performance improvement

---

### External_Benchmark

Evaluates IFOSS using an independent external test set.

The evaluation protocol follows the same benchmarking strategy while assessing model performance on unseen external data.

---

### umap_IFOSS_visualization.py

Produces supervised UMAP visualizations illustrating the effect of IFOSS preprocessing.

Visualization includes:

* Original training dataset
* IFOSS-preprocessed training dataset

---

### Simulation_ABLATION_study

Quantifies the contribution of each preprocessing component.

Compared strategies include:

* Baseline
* Isolation Forest
* One-Sided Selection
* Isolation Forest → One-Sided Selection (IFOSS)
* One-Sided Selection → Isolation Forest

Experiments are repeated across multiple random seeds and summarized using mean ± standard deviation.

---

### Simulation_noise_study

Evaluates the robustness of IFOSS under simulated data corruption.

Simulation scenarios include:

* Clean dataset
* Label noise
* Feature noise
* Class imbalance
* Combined corruption

Each scenario compares baseline CatBoost performance against IFOSS-preprocessed data.

---

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Input Data

The repository does not include any datasets.

Before running the scripts, load your dataset into memory as:

```python
X   # pandas DataFrame containing the feature matrix

y   # pandas Series or NumPy array containing binary class labels
```

Each script expects these variables to be available.

---

## Recommended Workflow

1. Load the dataset.
2. Run **IFOSS.py** to generate the resampled dataset.
3. Evaluate predictive performance using **Benchmark**.
4. Validate performance on an external dataset using **External_Benchmark**.
5. Visualize the preprocessing effect using **umap_IFOSS_visualization.py**.
6. Reproduce the supplementary experiments using:

   * **Simulation_ABLATION_study**
   * **Simulation_noise_study**

---

## Requirements

The implementation was tested using:

* Python 3.11
* NumPy
* pandas
* scikit-learn 1.5.2
* imbalanced-learn
* CatBoost
* XGBoost
* LightGBM
* Optuna
* UMAP
* Matplotlib

See `requirements.txt` for the complete list of dependencies.

---

## Reproducibility

Random seeds are fixed throughout the experiments to improve reproducibility.

The preprocessing, benchmarking, visualization, ablation, and simulation scripts correspond to the experiments reported in the accompanying manuscript.

---

## Citation

If you use this repository in your research, please cite the accompanying publication.

```text
Citation information will be added after publication.
```

---

## Contact

For questions, suggestions, or bug reports, please open a GitHub Issue.

---

## Acknowledgements

This repository accompanies the manuscript describing the IFOSS preprocessing framework for robust clinical machine learning.
