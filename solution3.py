import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Optional: convert to DataFrame for exploration
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
print(df.head())

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Train a Decision Tree Classifier
#    max_depth=3 to control overfitting a bit
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Check accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# (Optional) Visualize the tree structure
plt.figure(figsize=(8, 6))
plot_tree(model, filled=True)
plt.title('Arborele de Decizie - Structura')
plt.show()

# (Optional) Feature importances
feature_importances = model.feature_importances_
print("Importanța caracteristicilor:")
for feature, importance in zip(wine.feature_names, feature_importances):
    print(f"{feature}: {importance:.4f}")
