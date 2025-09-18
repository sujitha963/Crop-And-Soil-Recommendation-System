# 📊 Step 1: Import libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nicer
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# 📊 Step 2: Basic dataset info
print("Dataset Shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nMissing values:\n", data.isnull().sum())

# 📊 Step 3: Count of samples per crop
plt.figure(figsize=(12,6))
sns.countplot(y=data['label'], order=data['label'].value_counts().index)
plt.title("Number of Samples per Crop")
plt.xlabel("Count")
plt.ylabel("Crop Type")
plt.show()

# 📊 Step 4: Distribution of soil nutrients (N, P, K)
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.histplot(data['N'], bins=20, kde=True, ax=axes[0], color="green")
axes[0].set_title("Nitrogen (N) Distribution")
sns.histplot(data['P'], bins=20, kde=True, ax=axes[1], color="blue")
axes[1].set_title("Phosphorus (P) Distribution")
sns.histplot(data['K'], bins=20, kde=True, ax=axes[2], color="orange")
axes[2].set_title("Potassium (K) Distribution")
plt.tight_layout()
plt.show()

# 📊 Step 5: Distribution of temperature, humidity, pH, rainfall
fig, axes = plt.subplots(2, 2, figsize=(12,10))
sns.histplot(data['temperature'], bins=20, kde=True, ax=axes[0,0], color="red")
axes[0,0].set_title("Temperature Distribution")
sns.histplot(data['humidity'], bins=20, kde=True, ax=axes[0,1], color="purple")
axes[0,1].set_title("Humidity Distribution")
sns.histplot(data['ph'], bins=20, kde=True, ax=axes[1,0], color="brown")
axes[1,0].set_title("Soil pH Distribution")
sns.histplot(data['rainfall'], bins=20, kde=True, ax=axes[1,1], color="cyan")
axes[1,1].set_title("Rainfall Distribution")
plt.tight_layout()
plt.show()

# 📊 Step 6: Correlation heatmap between features
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap of Soil & Climate Features")
plt.show()

# 📊 Step 7: Boxplot of rainfall for each crop (to see requirements)
plt.figure(figsize=(14,6))
sns.boxplot(x="label", y="rainfall", data=data)
plt.xticks(rotation=90)
plt.title("Rainfall Requirement by Crop")
plt.show()

# 📊 Step 8: Pairplot for feature comparison (sample 500 for speed)
sns.pairplot(data.sample(500), hue="label", diag_kind="kde")
plt.suptitle("Pairplot of Features (Sample of 500)", y=1.02)
plt.show()
