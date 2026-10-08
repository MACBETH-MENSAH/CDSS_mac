import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

RANDOM_STATE = 42
NAVY = "#1E2761"; TEAL="#1B98B0"; LIGHT="#CADCFC"

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(1 - data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=2000, random_state=RANDOM_STATE),
}

# ---------- ROC CURVES (all models) ----------
plt.figure(figsize=(7,6))
colors = plt.cm.tab10(np.linspace(0,1,len(models)))
for (name, model), c in zip(models.items(), colors):
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=c, linewidth=2)
plt.plot([0,1],[0,1],"--", color="grey", linewidth=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Fig. 9.1: ROC Curves \u2013 All Candidate Models", fontsize=12, weight="bold", color=NAVY)
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("results/roc_curves.png", dpi=200)
plt.close()

# ---------- CONFUSION MATRICES (grid) ----------
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (name, model) in zip(axes.flat, models.items()):
    y_pred = model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14, weight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Benign","Malignant"]); ax.set_yticklabels(["Benign","Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(name, fontsize=11, weight="bold", color=NAVY)
plt.suptitle("Fig. 9.2: Confusion Matrices \u2013 All Candidate Models", fontsize=13, weight="bold", color=NAVY)
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=200)
plt.close()

# ---------- MODEL COMPARISON BAR CHART ----------
df = pd.read_csv("results/model_comparison.csv")
df = df.sort_values("F1", ascending=True)
fig, ax = plt.subplots(figsize=(9,5))
metrics = ["Accuracy","Precision","Recall","F1","ROC_AUC"]
colors2 = ["#8FA8DE","#5D7FC4","#2F5597","#1E2761","#0B1B4D"]
y_pos = np.arange(len(df))
bar_h = 0.15
for i, m in enumerate(metrics):
    ax.barh(y_pos + i*bar_h, df[m], height=bar_h, label=m, color=colors2[i])
ax.set_yticks(y_pos + bar_h*2)
ax.set_yticklabels(df["Model"])
ax.set_xlabel("Score")
ax.set_xlim(0.75, 1.02)
ax.set_title("Fig. 9.3: Model Performance Comparison", fontsize=12, weight="bold", color=NAVY)
ax.legend(loc="lower right", fontsize=8, ncol=1)
plt.tight_layout()
plt.savefig("results/model_comparison_chart.png", dpi=200)
plt.close()

# ---------- TOP FEATURES ----------
feat = pd.read_csv("results/top_features.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8,5))
feat.iloc[::-1].plot(kind="barh", legend=False, ax=ax, color=TEAL)
ax.set_xlabel("Importance"); ax.set_title("Fig. 9.4: Top 10 Most Important Features (Random Forest)", fontsize=11, weight="bold", color=NAVY)
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=200)
plt.close()

print("All plots saved to results/")
