import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

import warnings

warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('heart.csv')  # Replace with your CSV file path

# Assume the last column is the target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Define evaluation metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return accuracy, f1, recall, precision


# Create K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store results
results = []

# 1. Random Forest Classifier (transparent)
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X, y)
rf_best_params = grid_rf.best_params_

# Evaluate RF model
rf_accuracy, rf_f1, rf_recall, rf_precision = evaluate_model(grid_rf, X, y)
results.append(['Random Forest', rf_best_params, rf_accuracy, rf_f1, rf_recall, rf_precision])

# 2. Decision Tree Classifier (transparent, C4.5 approximation)
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, dt_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X, y)
dt_best_params = grid_dt.best_params_

# Evaluate DT model
dt_accuracy, dt_f1, dt_recall, dt_precision = evaluate_model(grid_dt, X, y)
results.append(['Decision Tree (C4.5)', dt_best_params, dt_accuracy, dt_f1, dt_recall, dt_precision])

# 3. Support Vector Machine (black-box)
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC(random_state=42)
grid_svm = GridSearchCV(svm, svm_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X, y)
svm_best_params = grid_svm.best_params_

# Evaluate SVM model
svm_accuracy, svm_f1, svm_recall, svm_precision = evaluate_model(grid_svm, X, y)
results.append(['SVM', svm_best_params, svm_accuracy, svm_f1, svm_recall, svm_precision])


# 4. Deep Learning Model (black-box)
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


dl_param_grid = {
    'epochs': [50, 100],
    'batch_size': [10, 20]
}
dl_model = KerasClassifier(build_fn=create_model, verbose=0)
grid_dl = GridSearchCV(dl_model, dl_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_dl.fit(X, y)
dl_best_params = grid_dl.best_params_

# Evaluate DL model
dl_accuracy, dl_f1, dl_recall, dl_precision = evaluate_model(grid_dl, X, y)
results.append(['Deep Learning', dl_best_params, dl_accuracy, dl_f1, dl_recall, dl_precision])

# Export results to CSV
results_df = pd.DataFrame(results, columns=['Model', 'Best Params', 'Accuracy', 'F1-Score', 'Recall', 'Precision'])
results_df.to_csv('model_results.csv', index=False)

print("Results saved to 'model_results.csv'")
