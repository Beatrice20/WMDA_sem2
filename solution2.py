import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# (Optional) Convert to a Pandas DataFrame for easier viewing
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
print(df.head())  # Uncomment to inspect

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a Naïve Bayes classifier (from Exercise 1)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 4. Train a Logistic Regression model
lr_model = LogisticRegression(solver='sag', max_iter=10000, random_state=42)
lr_model.fit(X_train, y_train)

# 5. Compare metrics: accuracy, precision, and recall for each model
# Note: Because we have three classes in the Wine dataset, we set average='macro' (or 'weighted') for multi-class
y_pred_nb = nb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

precision_nb = precision_score(y_test, y_pred_nb, average='macro')
precision_lr = precision_score(y_test, y_pred_lr, average='macro')

recall_nb = recall_score(y_test, y_pred_nb, average='macro')
recall_lr = recall_score(y_test, y_pred_lr, average='macro')

# 6. Print results
print(f"Naïve Bayes - Acuratețea: {accuracy_nb:.2f}, Precizia: {precision_nb:.2f}, Recall: {recall_nb:.2f}")
print(f"Logistic Regression - Acuratețea: {accuracy_lr:.2f}, Precizia: {precision_lr:.2f}, Recall: {recall_lr:.2f}")

# Optional: If you’d like to see a confusion matrix for each model
from sklearn.metrics import confusion_matrix
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

print("Confusion Matrix - Naïve Bayes:")
print(conf_matrix_nb)
print("Confusion Matrix - Logistic Regression:")
print(conf_matrix_lr)