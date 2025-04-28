import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# 1. Load the Dataset
df = pd.read_csv("heart.csv")  # Ensure heart.csv is in the same directory

# 2. Separate features (X) and target (y)
X = df.drop(columns=["target"])  # Assuming "target" is the column to predict
y = df["target"]

# 3. Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy before permutation: {accuracy * 100:.2f}%")

# 7. Compute Permutation Feature Importances
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# 8. Extract Feature Importances and Feature Names
feature_importances = perm_importance.importances_mean
feature_names = X.columns

# 9. Print Feature Importances
print("\nFeature Importances:")
for feature_name, importance in zip(feature_names, feature_importances):
    print(f"{feature_name}: {importance:.4f}")

# 10. Visualize Feature Importances
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Permutation Importance")
plt.title("Permutation Feature Importance (Heart Disease Dataset)")
plt.show()

# 10. Print Feature Importances in Tabular Format
print("\nFeature\t\tImportance")
sorted_idx = feature_importances.argsort()[::-1]  # Sort from high to low

for idx in sorted_idx:
    feature = feature_names[idx]
    importance = feature_importances[idx]
    print(f"{feature:<16}{importance:.3f}")
