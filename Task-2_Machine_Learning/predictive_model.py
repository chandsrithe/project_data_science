import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("../datasets/titanic.csv")

print("Original Shape:", df.shape)

# =====================================
# Data Cleaning
# =====================================

df = df.drop_duplicates()

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

df = df.drop(columns=["deck"])

# =====================================
# Encode Categorical Columns
# =====================================

encoder = LabelEncoder()

categorical_columns = [
    "sex",
    "embarked",
    "class",
    "who",
    "embark_town",
    "alive",
    "alone",
    "adult_male"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# =====================================
# Features & Target
# =====================================

X = df.drop("survived", axis=1)
y = df["survived"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Decision Tree
# =====================================

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, dt_pred))

# =====================================
# Random Forest
# =====================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

# =====================================
# Classification Report
# =====================================

print("\nClassification Report\n")
print(classification_report(y_test, rf_pred))

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# =====================================
# ROC Curve
# =====================================

rf_probs = rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, rf_probs)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# =====================================
# Feature Importance
# =====================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# =====================================
# Accuracy Comparison
# =====================================

accuracy_df = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ]
})

plt.figure(figsize=(6,4))
sns.barplot(data=accuracy_df, x="Model", y="Accuracy")
plt.ylim(0,1)
plt.title("Model Accuracy Comparison")
plt.savefig("model_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")
print("\nGenerated Files:")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- feature_importance.png")
print("- model_accuracy.png")