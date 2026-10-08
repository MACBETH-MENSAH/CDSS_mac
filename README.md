# Clinical Decision Support System Using Machine Learning: Breast Cancer Diagnostic Classification

A comparative study of six supervised machine learning algorithms on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, built as a proof-of-concept Clinical Decision Support System (CDSS).


> **Important:** This is an academic proof of concept. It is **not** a clinical tool and must not be used to make medical decisions. It was trained and evaluated on one clean public research dataset, with no real hospital data and no clinical validation.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Results at a glance](#results-at-a-glance)
3. [Repository structure](#repository-structure)
4. [Quick start](#quick-start)
5. [What the code does](#what-the-code-does)
6. [Dataset](#dataset)
7. [Methodology summary](#methodology-summary)
8. [Reproducibility](#reproducibility)
9. [Limitations](#limitations)
10. [Future work](#future-work)
11. [References](#references)
12. [Group members](#group-members)

---

## What this project does

Many published studies on clinical prediction report one algorithm's accuracy without a controlled comparison, treat interpretability as secondary, and rely on a single train/test split. This project addresses those three gaps by:

- training **six** different classifiers under **one identical pipeline**,
- evaluating them with **five metrics** (accuracy, precision, recall, F1, ROC-AUC) plus **5-fold cross-validation**,
- extracting **feature importances** from the Random Forest to check which clinical measurements drive the predictions.

The models are Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (RBF kernel), Naïve Bayes, and a Neural Network (MLP).

---

## Results at a glance

Held-out test set: 114 patients (72 benign, 42 malignant). Malignant is the positive class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV accuracy |
|---|---|---|---|---|---|---|
| SVM (RBF) | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 | 0.971 ± 0.005 |
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 | 0.974 ± 0.015 |
| Neural Network (MLP) | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 | 0.976 ± 0.013 |
| Random Forest | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 | 0.960 ± 0.016 |
| Naïve Bayes | 0.921 | 0.923 | 0.857 | 0.889 | 0.989 | 0.938 ± 0.026 |
| Decision Tree | 0.921 | 0.946 | 0.833 | 0.886 | 0.945 | 0.908 ± 0.040 |

**Key takeaways**

- SVM scored highest on test accuracy and F1. Logistic Regression had the highest ROC-AUC.
- The top four models are within about one percentage point of each other, so the ranking among them is not decisive.
- Every model missed more malignant cases (false negatives) than it falsely flagged. For a diagnostic tool, recall matters more than headline accuracy.
- Top Random Forest features: worst perimeter, worst area, worst concave points, mean concave points, worst radius. All describe tumour size and nuclear shape irregularity.

The exact numbers are in [`results/model_comparison.csv`](results/model_comparison.csv).

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py            # trains and evaluates all six models, saves results
├── make_result_plots.py       # generates the figures from the saved results
└── results/
    ├── model_comparison.csv   # full metrics table (all models)
    ├── top_features.csv       # top 10 Random Forest feature importances
    ├── summary.json           # dataset summary and best model
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── model_comparison_chart.png
    └── feature_importance.png
```

---

## Quick start

**Requirements:** Python 3.9 or newer, and `git`.

```bash
# 1. Clone the repository
git clone [https://github.com/MACBETH-MENSAH/CDSS_mac.git]
cd CDSS_mac

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train and evaluate the models (prints the results table)
python3 run_pipeline.py

# 5. Generate the figures into results/
python3 make_result_plots.py
```

Run the two scripts **in this order**. `make_result_plots.py` reads the CSV files that `run_pipeline.py` writes.

**No dataset download is needed.** The data loads from inside scikit-learn.

**Google Colab / Jupyter:** paste the contents of each script into a notebook cell (after `!pip install -r requirements.txt` or `!pip install scikit-learn pandas matplotlib seaborn`) and run them in the same order.

---

## What the code does

### `run_pipeline.py`

1. Loads the WDBC dataset with `sklearn.datasets.load_breast_cancer()`.
2. Recodes the label so that **1 = malignant** (the positive class).
3. Splits the data 80/20 with stratification (`random_state=42`).
4. Standardises all 30 features with `StandardScaler`.
5. Trains six models: Logistic Regression, Decision Tree (max depth 5), Random Forest (300 trees, max depth 8), SVM (RBF), Gaussian Naïve Bayes, and an MLP (hidden layers 32 and 16).
6. Evaluates each on the test set: accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.
7. Runs 5-fold stratified cross-validation for each model.
8. Extracts Random Forest feature importances.
9. Saves everything to `results/`.

### `make_result_plots.py`

Retrains the models with the same settings and saves four figures to `results/`: ROC curves for all models, a grid of confusion matrices, a grouped bar chart of all metrics, and the top-10 feature importance chart.

---

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**

- 569 patient records, 30 numeric features, binary outcome (malignant or benign).
- Class split: 357 benign (62.7%), 212 malignant (37.3%). No missing values.
- Each feature is computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension, each as a mean, standard error and "worst" value.
- **Source:** the original dataset was created by Street, Wolberg and Mangasarian (1993) and is hosted by the UCI Machine Learning Repository. This project loads the identical data through scikit-learn's bundled copy.

**On the dataset's age:** the data dates from 1993, but it remains a widely used benchmark, and recent studies (2018, 2021 and 2025 among those we reviewed) still validate methods against it. The aim of this project is rigorous algorithm comparison, not dataset novelty. The limits of using an older, clean benchmark are acknowledged below.

---

## Methodology summary

```
Load WDBC (569 x 30)
        |
Stratified 80/20 train/test split
        |
Standardise features (StandardScaler)
        |
Train six models on identical data
        |
5-fold stratified cross-validation
        |
Evaluate on held-out test set
(accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
        |
Compare models + Random Forest feature importance
```

**Why these six models:** they cover the main families of supervised classifiers (linear, tree, tree ensemble, margin-based, probabilistic, neural), so the comparison is between fundamentally different approaches.

**Why several metrics:** accuracy alone hides the type of error. Recall shows how many malignant cases were missed, which is the clinically costly mistake.

---

## Reproducibility

- A fixed random seed (`42`) is used for the split, the models and the cross-validation.
- Re-running on the same library versions gives identical results.
- Across different scikit-learn or Python versions, small differences in the last decimal places are possible (particularly for the Random Forest and the Neural Network). The overall ranking and conclusions should hold.
- Record your own versions when reproducing: `python --version` and `pip show scikit-learn`.

---

## Limitations

- **One clean public dataset.** No real hospital data, and nothing tested on Ghanaian or African patient populations.
- **Small test set.** 114 patients, 42 malignant. One extra missed case changes recall by about 2.4 points, so differences among the top four models are within noise.
- **Limited hyperparameter tuning.** Models use reasonable defaults or light adjustments, not an exhaustive search.
- **Scaling and cross-validation.** The scaler is fitted on the full training set before cross-validation, so scaling statistics leak slightly into the validation folds. The effect is small, but a stricter design would place scaling inside a `Pipeline` used for cross-validation.
- **No clinical validation.** No clinician reviewed the outputs or the feature-importance results.
- **Basic explainability.** Random Forest feature importance is global (it explains the model overall, not a single patient's prediction). Correlated features such as radius, perimeter and area share importance between them.

---

## Future work

- Apply the same pipeline to a larger, locally sourced clinical dataset.
- Add SHAP for per-prediction explanations.
- Run a systematic hyperparameter search (grid or Bayesian).
- Move scaling inside a cross-validation `Pipeline`.
- Add fairness analysis across patient subgroups where demographic data is available.
- Seek clinician review of the model outputs.

---

## References

**Dataset source**

- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE International Symposium on Electronic Imaging: Science and Technology*, 1905, 861–870.

**Reviewed literature (2018–2025)**

1. Agarap, A. F. M. (2018). On breast cancer detection: An application of machine learning algorithms on the Wisconsin Diagnostic Dataset. arXiv:1711.07831.
2. Entezari, R. (2018). Breast cancer diagnosis via classification algorithms. arXiv:1807.01334.
3. A fast and interpretable logistic regression framework for breast tumor classification using the Wisconsin Diagnostic Dataset (2025). medRxiv preprint.
4. Advanced deep learning and transfer learning approaches for breast cancer classification (2025). PubMed Central.
5. Yedjou, C. G., Tchounwou, S. S., Aló, R. A., Elhag, R., Mochona, B., & Latinwo, L. (2021). Application of machine learning algorithms in breast cancer diagnosis and classification. *International Journal of Scientific Academic Research*, 2(1), 3081–3086.
6. Naji, M. A., El Filali, S., Aarika, K., Benlahmar, E. H., Ait Abdelouhahid, R., & Debauche, O. (2021). Machine learning algorithms for breast cancer prediction and diagnosis. *Procedia Computer Science*, 191, 487–492.
7. Chaddad, A., Lu, Q., Li, J., Katib, Y., & Kateb, R. (2024). Explainable artificial intelligence for medical applications: A review. arXiv:2412.01829.
8. A review on explainable artificial intelligence for healthcare (2023). arXiv:2304.04780.
9. Band, S. S., Yarahmadi, A., Hsu, C.-C., Biyari, M., Sookhak, M., Ameri, R., Dehzangi, I., Chronopoulos, A. T., & Liang, H.-W. (2023). Application of explainable artificial intelligence in medical health: A systematic review of interpretability methods. *Informatics in Medicine Unlocked*, 40, 101286.
10. Ndembi, N., Rammer, B., Fokam, J., et al. (2025). Integrating artificial intelligence into African health systems and emergency response: Need for an ethical framework and guidelines. *Journal of Public Health in Africa*, 16(1).

---

## Disclaimer

This repository is for educational purposes. The models are not validated for clinical use, and nothing here constitutes medical advice.
--Developed by: Macbeth Mensah(BSc. Computer Engineering)
