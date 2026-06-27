import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("../datasets/Sample - Superstore.csv", encoding="latin1")

print("="*50)
print("REAL WORLD DATA PROJECT - SUPERSTORE")
print("="*50)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# ==========================================
# Data Cleaning
# ==========================================

df.drop_duplicates(inplace=True)

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create Month-Year column
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

print("\nShape After Cleaning:")
print(df.shape)

# ==========================================
# Visualization 1
# Sales by Category
# ==========================================

plt.figure(figsize=(8,5))
sns.barplot(
    x=df.groupby("Category")["Sales"].sum().index,
    y=df.groupby("Category")["Sales"].sum().values
)
plt.title("Total Sales by Category")
plt.savefig("sales_by_category.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 2
# Profit by Category
# ==========================================

plt.figure(figsize=(8,5))
sns.barplot(
    x=df.groupby("Category")["Profit"].sum().index,
    y=df.groupby("Category")["Profit"].sum().values
)
plt.title("Profit by Category")
plt.savefig("profit_by_category.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 3
# Sales by Region
# ==========================================

plt.figure(figsize=(8,5))
sns.barplot(
    x=df.groupby("Region")["Sales"].sum().index,
    y=df.groupby("Region")["Sales"].sum().values
)
plt.title("Sales by Region")
plt.savefig("sales_by_region.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# Visualization 4
# Monthly Sales Trend
# ==========================================

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(12,5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales.png", dpi=300)
plt.show()
plt.close()

# ==========================================
# Visualization 5
# Profit Distribution
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["Profit"], bins=30, kde=True)
plt.title("Profit Distribution")
plt.savefig("profit_distribution.png", dpi=300)
plt.show()
plt.close()

# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(8,6))
sns.heatmap(
    df[["Sales","Quantity","Discount","Profit"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap_retail.png", dpi=300)
plt.show()
plt.close()

# ==========================================
# Machine Learning
# Predict Sales
# ==========================================

X = df[["Quantity","Discount","Profit"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Performance")
print("-"*30)

print("MAE :", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2 Score:", r2_score(y_test, predictions))

# ==========================================
# Prediction Plot
# ==========================================

plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig("sales_prediction.png", dpi=300)
plt.show()
plt.close()

print("\nPROJECT COMPLETED SUCCESSFULLY!")

print("\nGenerated Files:")
print("- sales_by_category.png")
print("- profit_by_category.png")
print("- sales_by_region.png")
print("- monthly_sales.png")
print("- profit_distribution.png")
print("- correlation_heatmap_retail.png")
print("- sales_prediction.png")