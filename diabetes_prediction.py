# ============================================================
# TASK 4: Disease Prediction from Medical Data
# Dataset: Pima Indians Diabetes Dataset
# Models: Logistic Regression, SVM, Random Forest, XGBoost
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
# Option A: Load from file (if you have it downloaded)
# df = pd.read_csv("diabetes.csv")

# Option B: Load directly from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, names=columns)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass Distribution:\n{df['Outcome'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe()}")

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

# In this dataset, 0 values in medical columns = missing data
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())  # Fill with median

print("\n" + "=" * 50)
print("AFTER HANDLING MISSING VALUES (zeros replaced with median)")
print("=" * 50)
print(df.isnull().sum())

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (important for SVM & Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ============================================================
# STEP 3: TRAIN MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

results = {}

print("\n" + "=" * 50)
print("MODEL TRAINING & EVALUATION")
print("=" * 50)

for name, model in models.items():
    # Use scaled data for LR and SVM, raw for tree-based
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    cv    = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    results[name] = {"Accuracy": acc, "ROC-AUC": auc, "CV Accuracy": cv,
                     "model": model, "y_pred": y_pred, "y_prob": y_prob}

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  CV Accuracy : {cv:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

# ============================================================
# STEP 4: COMPARISON TABLE
# ============================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
summary = pd.DataFrame({
    name: {
        "Accuracy":    f"{v['Accuracy']:.4f}",
        "ROC-AUC":     f"{v['ROC-AUC']:.4f}",
        "CV Accuracy": f"{v['CV Accuracy']:.4f}"
    }
    for name, v in results.items()
}).T
print(summary)

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Diabetes Prediction — Model Comparison", fontsize=16, fontweight="bold")

# --- 5a: Accuracy Bar Chart ---
ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies  = [results[m]["Accuracy"] for m in model_names]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
bars = ax1.bar(model_names, accuracies, color=colors, edgecolor="black", width=0.5)
ax1.set_ylim(0.6, 1.0)
ax1.set_title("Model Accuracy Comparison")
ax1.set_ylabel("Accuracy")
ax1.set_xticklabels(model_names, rotation=15)
for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10)

# --- 5b: ROC Curves ---
ax2 = axes[0, 1]
for name, color in zip(model_names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
    ax2.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC-AUC']:.3f})", color=color)
ax2.plot([0,1],[0,1],"k--", label="Random")
ax2.set_title("ROC-AUC Curves")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=8)

# --- 5c: Confusion Matrix (Best Model) ---
best_model_name = max(results, key=lambda m: results[m]["ROC-AUC"])
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"])
ax3.set_title(f"Confusion Matrix — {best_model_name}")
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")

# --- 5d: Feature Importance (Random Forest) ---
ax4 = axes[1, 1]
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind="barh", ax=ax4, color="#55A868", edgecolor="black")
ax4.set_title("Feature Importance — Random Forest")
ax4.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("diabetes_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: diabetes_results.png")

# ============================================================
# STEP 6: SAVE BEST MODEL
# ============================================================

best_model = results[best_model_name]["model"]
joblib.dump(best_model, "best_diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\n✅ Best Model : {best_model_name} (AUC: {results[best_model_name]['ROC-AUC']:.4f})")
print("✅ Model saved : best_diabetes_model.pkl")
print("✅ Scaler saved: scaler.pkl")

# ============================================================
# STEP 7: PREDICT ON NEW PATIENT (Example)
# ============================================================

print("\n" + "=" * 50)
print("SAMPLE PREDICTION ON NEW PATIENT")
print("=" * 50)

new_patient = pd.DataFrame([{
    "Pregnancies": 3,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}])

if best_model_name in ["Logistic Regression", "SVM"]:
    new_scaled = scaler.transform(new_patient)
    prediction = best_model.predict(new_scaled)[0]
    probability = best_model.predict_proba(new_scaled)[0][1]
else:
    prediction = best_model.predict(new_patient)[0]
    probability = best_model.predict_proba(new_patient)[0][1]

print(f"Patient Data   : {new_patient.iloc[0].to_dict()}")
print(f"Prediction     : {'⚠️  DIABETIC' if prediction == 1 else '✅ NOT DIABETIC'}")
print(f"Probability    : {probability:.2%}")
