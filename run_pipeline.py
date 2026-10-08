import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve, classification_report)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# 1. LOAD DATASET  (Wisconsin Diagnostic Breast Cancer - real, public,
#    clinical dataset; malignant=0 / benign=1 originally, we relabel
#    so 1 = malignant/at-risk, matching a clinical "positive" case)
# ------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(1 - data.target, name="malignant")  # 1 = malignant (positive/at-risk class)

print("Dataset shape:", X.shape)
print("Class balance:\n", y.value_counts())
print("Malignant (positive) rate: {:.1f}%".format(100 * y.mean()))

# ------------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3. MODELS
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=RANDOM_STATE),
}

results = []
roc_data = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    results.append({
        "Model": name,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": auc,
        "CV_Mean_Acc": cv_scores.mean(), "CV_Std": cv_scores.std(),
        "TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1]),
    })
    print(f"\n--- {name} ---")
    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
results_df.to_csv("results/model_comparison.csv", index=False)
print("\n\n=== FINAL COMPARISON TABLE (sorted by F1) ===")
print(results_df[["Model","Accuracy","Precision","Recall","F1","ROC_AUC"]].round(4).to_string(index=False))

# ------------------------------------------------------------------
# 4. FEATURE IMPORTANCE (Random Forest) - simple explainability
# ------------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top10 = importances.head(10)
print("\nTop 10 most important features (Random Forest):")
print(top10.round(4).to_string())
top10.to_csv("results/top_features.csv")

with open("results/summary.json", "w") as f:
    json.dump({
        "dataset": "Wisconsin Diagnostic Breast Cancer (sklearn built-in, UCI ML Repository)",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "positive_rate_pct": float(round(100*y.mean(),2)),
        "best_model_by_f1": results_df.iloc[0]["Model"],
    }, f, indent=2)

print("\nSaved: results/model_comparison.csv, results/top_features.csv, results/summary.json")
