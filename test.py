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


def runRF(name, data):
    try:
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
            'n_estimators': [500, 1000],  # Two values, balancing performance and training time
            'max_depth': [9, 15, None],  # Includes a constrained value, a mid-range, and unlimited depth
            'min_samples_split': [5, 10],  # Slightly higher values to prevent overfitting on large data
            'min_samples_leaf': [1, 2],  # Regularizing by ensuring a minimum number of samples in leaves
            'max_features': ['auto', 'sqrt']  # Two commonly used options, balancing speed and accuracy
        }
        rf = RandomForestClassifier(random_state=42)
        grid_rf = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
        grid_rf.fit(X_train, y_train)
        rf_best_params = grid_rf.best_params_

        # chart with the most important features
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

        print(f"{name}: {rf_accuracy} accuracy")
        print(f"{name}: {rf_recall} recall")
        print(f"{name}: {rf_precision} precision")
        print(f"{name}: {rf_f1} F1")
    except Exception as e:
        print(e)


def runDT(name, data):
    try:
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
        dt_param_grid = {
            'criterion': ['gini'],
            'max_depth': [7],
            'min_samples_split': [5]
        }
        dt = DecisionTreeClassifier(random_state=42)
        grid_dt = GridSearchCV(dt, dt_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
        grid_dt.fit(X_train, y_train)

        # Evaluate DT model
        dt_accuracy, dt_f1, dt_recall, dt_precision = evaluate_model(grid_dt, X_test, y_test)

        print(f"{name}: {dt_accuracy} accuracy")
        print(f"{name}: {dt_recall} recall")
        print(f"{name}: {dt_precision} precision")
        print(f"{name}: {dt_f1} F1")
    except Exception as e:
        print(e)

def run_rf_dropping_attributes(name, data):
    try:
        print('Dataset loaded.')

        # Preserve the 'hasHeadache' column (target)
        hasHeadache_col = data.pop('hasHeadache')
        data['hasHeadache'] = hasHeadache_col

        # Keep X as a DataFrame, no need to use .values here
        X = data.drop(columns=['hasHeadache'])  # Exclude the target column
        y = data['hasHeadache']  # Target column

        # Create K-Fold cross-validator
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Dictionary to store results
        results = []

        # Random Forest parameter grid
        rf_param_grid = {
            'n_estimators': [1000],
            'max_depth': [9],
            'min_samples_split': [5]
        }

        # Loop through each attribute in X to drop one at a time
        for attribute in X.columns:
            # Create a new dataset by dropping one attribute
            X_dropped = X.drop(columns=[attribute])

            # Split the modified dataset
            X_train_dropped, X_test_dropped, y_train, y_test = train_test_split(
                X_dropped, y, test_size=0.3, random_state=42
            )

            # Train Random Forest Classifier
            rf = RandomForestClassifier(random_state=42)
            grid_rf = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
            grid_rf.fit(X_train_dropped, y_train)

            # Evaluate the model with the dropped attribute
            rf_accuracy, rf_f1, rf_recall, rf_precision = evaluate_model(grid_rf, X_test_dropped, y_test)

            # Print the results for the dropped attribute
            print(f"{name} (dropped {attribute}): {rf_accuracy} accuracy")
            print(f"{name} (dropped {attribute}): {rf_recall} recall")
            print(f"{name} (dropped {attribute}): {rf_precision} precision")
            print(f"{name} (dropped {attribute}): {rf_f1} F1")

    except Exception as e:
        print(e)
