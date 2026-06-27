import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("titanic.csv")

print("Original Shape:", df.shape)

# ==========================================
# Data Cleaning
# ==========================================

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# Drop deck column because it has many missing values
df = df.drop(columns=["deck"])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nNew Shape:", df.shape)

# ==========================================
# Visualization 1: Survival Count
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="survived")
plt.title("Passenger Survival Count")
plt.savefig("survival_count.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 2: Gender vs Survival
# ==========================================

plt.figure(figsize=(7,5))
sns.countplot(data=df, x="sex", hue="survived")
plt.title("Gender vs Survival")
plt.savefig("gender_vs_survival.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 3: Passenger Class
# ==========================================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="pclass")
plt.title("Passenger Class Distribution")
plt.savefig("passenger_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 4: Age Distribution
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 5: Fare Outliers
# ==========================================

plt.figure(figsize=(8,5))
sns.boxplot(x=df["fare"])
plt.title("Fare Outliers")
plt.savefig("fare_outliers.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Correlation Heatmap
# ==========================================

df_corr = df.copy()

df_corr["sex"] = df_corr["sex"].map({
    "male": 0,
    "female": 1
})

plt.figure(figsize=(10,6))
sns.heatmap(
    df_corr.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\n====================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("====================================")
print("Images saved in the current folder:")
print("1. survival_count.png")
print("2. gender_vs_survival.png")
print("3. passenger_class_distribution.png")
print("4. age_distribution.png")
print("5. fare_outliers.png")
print("6. correlation_heatmap.png")