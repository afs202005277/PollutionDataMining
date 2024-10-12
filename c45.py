from C45 import C45Classifier
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import sys


data = pd.read_csv('data_processed.csv')
X = data.drop(['hasHeadache'], axis=1)
y = data['hasHeadache']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Perform KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = C45Classifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    accuracies.append(accuracy)


mean_accuracy = sum(accuracies) / len(accuracies)
print(f'MEAN ACCURACY: {mean_accuracy:.3f}')

sys.stdout = open('c45rules.txt', 'wt')
model.print_rules()
