import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("titanic.csv")

print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# ==========================================
# Data Cleaning
# ==========================================

df = df.drop_duplicates()

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

df = df.drop(columns=["deck"])

# ==========================================
# Plot 1 - Survival Count
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="survived")
plt.title("Passenger Survival Count")
plt.savefig("eda_survival.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Plot 2 - Survival by Gender
# ==========================================

plt.figure(figsize=(7,5))
sns.countplot(data=df, x="sex", hue="survived")
plt.title("Survival by Gender")
plt.savefig("eda_gender.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Plot 3 - Passenger Class
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="pclass")
plt.title("Passenger Class Distribution")
plt.savefig("eda_class.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Plot 4 - Age Distribution
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("eda_age.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Plot 5 - Fare Distribution
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.savefig("eda_fare.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Correlation Heatmap
# ==========================================

df_corr = df.copy()

categorical = [
    "sex",
    "embarked",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alive",
    "alone"
]

for col in categorical:
    df_corr[col] = df_corr[col].astype("category").cat.codes

plt.figure(figsize=(10,7))
sns.heatmap(
    df_corr.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("eda_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Pairplot
# ==========================================

pair = sns.pairplot(
    df[["survived","age","fare","pclass"]],
    hue="survived"
)

pair.savefig("eda_pairplot.png", dpi=300)

plt.close()

# ==========================================
# Observations
# ==========================================

print("\nKey Observations")
print("-"*40)
print("1. Most passengers belonged to Third Class.")
print("2. Female passengers had a higher survival rate.")
print("3. Most passengers were between 20 and 40 years old.")
print("4. Fare distribution is right-skewed.")
print("5. Survival is influenced by passenger class and gender.")

print("\nEDA Completed Successfully!")
print("\nGenerated Files:")
print("- eda_survival.png")
print("- eda_gender.png")
print("- eda_class.png")
print("- eda_age.png")
print("- eda_fare.png")
print("- eda_heatmap.png")
print("- eda_pairplot.png")