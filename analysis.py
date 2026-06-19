import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Load Dataset
# ======================

df = pd.read_csv("titanic.csv")

print("Original Shape:", df.shape)

# ======================
# Data Cleaning
# ======================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["embark_town"].fillna(df["embark_town"].mode()[0], inplace=True)

# Drop deck column
df.drop("deck", axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nNew Shape:", df.shape)

# ======================
# Visualization 1
# ======================

plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=df)
plt.title("Passenger Survival Count")
plt.show()

# ======================
# Visualization 2
# ======================

plt.figure(figsize=(7,5))
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Gender vs Survival")
plt.show()

# ======================
# Visualization 3
# ======================

plt.figure(figsize=(6,4))
sns.countplot(x="pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

# ======================
# Visualization 4
# ======================

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# ======================
# Visualization 5
# ======================

plt.figure(figsize=(8,5))
sns.boxplot(x=df["fare"])
plt.title("Fare Outliers")
plt.show()

# ======================
# Heatmap
# ======================

df_corr = df.copy()

df_corr["sex"] = df_corr["sex"].map({
    "male": 0,
    "female": 1
})

plt.figure(figsize=(10,6))
sns.heatmap(
    df_corr.select_dtypes(include="number").corr(),
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()