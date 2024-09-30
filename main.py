import pandas as pd
import numpy as np
from scipy.stats import describe
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

def convert_integer_columns(df):
    # Iterate through each column in the DataFrame
    for col in df.select_dtypes(include=['int']):
        # Get the minimum and maximum values of the column
        min_val = df[col].min()
        max_val = df[col].max()

        # Determine the appropriate integer type
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)  # This covers any larger integers

    return df

def convert_float_columns(df):
    # Iterate through each column in the DataFrame
    for col in df.select_dtypes(include=['float']):
        # Get the minimum and maximum values of the column
        min_val = df[col].min()
        max_val = df[col].max()

        # Determine the appropriate float type
        if min_val >= np.finfo(np.float16).min and max_val <= np.finfo(np.float16).max:
            df[col] = df[col].astype(np.float16)
        elif min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype(np.float64)  # This covers any larger floats

    return df

def check_non_float_columns(df):
    non_float_columns = []

    for column in df.columns:
        print(column)
        for value in df[column]:
            try:
                float(value)  # Try to convert to float
            except ValueError:
                print(value)
                non_float_columns.append(column)
                break  # No need to check further if one non-convertible value is found

    return list(set(non_float_columns))

# Define evaluation metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return accuracy, f1, recall, precision

# Load dataset
data = pd.read_csv('data_processed.csv')[:100000]  # Replace with your CSV file path
#data = convert_integer_columns(data)
#data = convert_float_columns(data)
print("Hello!")
hasHeadache_col = data.pop('hasHeadache')
data['hasHeadache'] = hasHeadache_col

print(list(data.columns)[-1])

# Assume the last column is the target
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
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
rf_best_params = grid_rf.best_params_

# Evaluate RF model
rf_accuracy, rf_f1, rf_recall, rf_precision = evaluate_model(grid_rf, X_test, y_test)
results.append(['Random Forest', rf_best_params, rf_accuracy, rf_f1, rf_recall, rf_precision])

print("Finished random forest")

# 2. Decision Tree Classifier (transparent, C4.5 approximation)
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, dt_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train, y_train)
dt_best_params = grid_dt.best_params_

# Evaluate DT model
dt_accuracy, dt_f1, dt_recall, dt_precision = evaluate_model(grid_rf, X_test, y_test)
results.append(['Decision Tree (C4.5)', dt_best_params, dt_accuracy, dt_f1, dt_recall, dt_precision])

print("Finished DT")

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
#grid_dl.fit(X_train, y_train)
#dl_best_params = grid_dl.best_params_

# Evaluate DL model
#dl_accuracy, dl_f1, dl_recall, dl_precision = evaluate_model(grid_rf, X_test, y_test)
#results.append(['Deep Learning', dl_best_params, dl_accuracy, dl_f1, dl_recall, dl_precision])

print("Finished DL")

# Export results to CSV
results_df = pd.DataFrame(results, columns=['Model', 'Best Params', 'Accuracy', 'F1-Score', 'Recall', 'Precision'])
results_df.to_csv('model_results.csv', index=False)

print("Results saved to 'model_results.csv'")
