import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, \
    SGDClassifier, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Binarizer, binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Define evaluation metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return accuracy, f1, recall, precision

# Load dataset
data = pd.read_csv('sample.csv')

print('Dataset loaded.')

hasHeadache_col = data.pop('hasHeadache')
data['hasHeadache'] = hasHeadache_col

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store results
results = []

# 1. Random Forest Classifier (transparent)
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 7, 9],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
rf_best_params = grid_rf.best_params_

#chart with the most important features
importances = grid_rf.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(20, 10))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), data.columns[:-1][indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.savefig('feature_importances.png', format='png')

# Evaluate RF model
rf_accuracy, rf_f1, rf_recall, rf_precision = evaluate_model(grid_rf, X_test, y_test)
results.append(['Random Forest', rf_best_params, rf_accuracy, rf_f1, rf_recall, rf_precision])

print("Finished random forest")

# Export results to CSV
results_df = pd.DataFrame(results, columns=['Model', 'Best Params', 'Accuracy', 'F1-Score', 'Recall', 'Precision'])
results_df.to_csv('model_results.csv', index=False)

print("Results saved to 'model_results.csv'")