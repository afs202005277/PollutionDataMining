import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
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
data = pd.read_csv('data_processed.csv')[:1000]

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

# 2. Decision Tree Classifier (transparent, C4.5 approximation)
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 7, 9],
    'min_samples_split': [2, 5, 10]
}
dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, dt_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train, y_train)
dt_best_params = grid_dt.best_params_

# Evaluate DT model
dt_accuracy, dt_f1, dt_recall, dt_precision = evaluate_model(grid_dt, X_test, y_test)
results.append(['Decision Tree (C4.5)', dt_best_params, dt_accuracy, dt_f1, dt_recall, dt_precision])

print("Finished DT")

# Save the best decision tree
best_dt_model = grid_dt.best_estimator_
plt.figure(figsize=(50, 50))
plot_tree(best_dt_model, filled=True, feature_names=data.columns[:-1], class_names=["0", "1"], rounded=True)
plt.savefig('best_decision_tree.png', format='png')
print("Decision tree saved as 'best_decision_tree.png'")

# 4. Deep Learning Model (black-box)
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

epochs = 50
batch_size = 32
dl_model = create_model()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

dl_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)


# Evaluate DL model
y_pred = dl_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print(y_pred)

print(y_test)


dl_accuracy = accuracy_score(y_test, y_pred)
dl_f1 = f1_score(y_test, y_pred)
dl_recall = recall_score(y_test, y_pred)
dl_precision = precision_score(y_test, y_pred)
results.append(['Deep Learning', {f"'epochs': {epochs}, 'batch_size': {batch_size}"}, dl_accuracy, dl_f1, dl_recall, dl_precision])

print("Finished DL")

# Export results to CSV
results_df = pd.DataFrame(results, columns=['Model', 'Best Params', 'Accuracy', 'F1-Score', 'Recall', 'Precision'])
results_df.to_csv('model_results.csv', index=False)

print("Results saved to 'model_results.csv'")
