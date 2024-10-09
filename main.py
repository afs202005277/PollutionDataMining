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

def runTests(csv_name, results_name):

    # Load dataset
    data = pd.read_csv(f'{csv_name}')

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

    # 3. K-Nearest Neighbors (KNN)
    knn_param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
    }
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, knn_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    knn_best_params = grid_knn.best_params_

    # Evaluate KNN model
    knn_accuracy, knn_f1, knn_recall, knn_precision = evaluate_model(grid_knn, X_test, y_test)
    results.append(['KNN', knn_best_params, knn_accuracy, knn_f1, knn_recall, knn_precision])

    print("Finished KNN")

    # 4. Logistic Regression
    lr_param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 200, 500]
    }
    lr = LogisticRegression(random_state=42)
    grid_lr = GridSearchCV(lr, lr_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    lr_best_params = grid_lr.best_params_

    # Evaluate LR model
    lr_accuracy, lr_f1, lr_recall, lr_precision = evaluate_model(grid_lr, X_test, y_test)
    results.append(['Logistic Regression', lr_best_params, lr_accuracy, lr_f1, lr_recall, lr_precision])

    print("Finished LR")

    # 5. Gradient Boosting Classifier
    gb_param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    }
    gb = GradientBoostingClassifier(random_state=42)
    grid_gb = GridSearchCV(gb, gb_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    gb_best_params = grid_gb.best_params_

    # Evaluate GB model
    gb_accuracy, gb_f1, gb_recall, gb_precision = evaluate_model(grid_gb, X_test, y_test)
    results.append(['Gradient Boosting', gb_best_params, gb_accuracy, gb_f1, gb_recall, gb_precision])

    print("Finished GB")

    # 6. AdaBoost Classifier
    ada_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    ada = AdaBoostClassifier(random_state=42)
    grid_ada = GridSearchCV(ada, ada_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_ada.fit(X_train, y_train)
    ada_best_params = grid_ada.best_params_

    # Evaluate AdaBoost model
    ada_accuracy, ada_f1, ada_recall, ada_precision = evaluate_model(grid_ada, X_test, y_test)
    results.append(['AdaBoost', ada_best_params, ada_accuracy, ada_f1, ada_recall, ada_precision])

    print("Finished AdaBoost")

    # 8. Naive Bayes (GaussianNB)
    gnb_param_grid = {}  # No hyperparameters to tune for GaussianNB
    gnb = GaussianNB()
    grid_gnb = GridSearchCV(gnb, gnb_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_gnb.fit(X_train, y_train)

    # Evaluate Naive Bayes model
    gnb_accuracy, gnb_f1, gnb_recall, gnb_precision = evaluate_model(grid_gnb, X_test, y_test)
    results.append(['GaussianNB', None, gnb_accuracy, gnb_f1, gnb_recall, gnb_precision])

    print("Finished GaussianNB")

    # 10. Extra Trees Classifier
    et_param_grid = {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 7, 9],
        'min_samples_split': [2, 5, 10]
    }
    et = ExtraTreesClassifier(random_state=42)
    grid_et = GridSearchCV(et, et_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_et.fit(X_train, y_train)
    et_best_params = grid_et.best_params_

    # Evaluate Extra Trees model
    et_accuracy, et_f1, et_recall, et_precision = evaluate_model(grid_et, X_test, y_test)
    results.append(['Extra Trees', et_best_params, et_accuracy, et_f1, et_recall, et_precision])

    print("Finished Extra Trees")


    # 11. Multilayer Perceptron (Neural Network)
    mlp_param_grid = {
        'hidden_layer_sizes': [(50,50), (100,), (150,50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp = MLPClassifier(random_state=42, max_iter=500)
    grid_mlp = GridSearchCV(mlp, mlp_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_mlp.fit(X_train, y_train)
    mlp_best_params = grid_mlp.best_params_

    # Evaluate MLP model
    mlp_accuracy, mlp_f1, mlp_recall, mlp_precision = evaluate_model(grid_mlp, X_test, y_test)
    results.append(['MLP', mlp_best_params, mlp_accuracy, mlp_f1, mlp_recall, mlp_precision])

    print("Finished MLP")


    # 13. Bagging Classifier
    bag_param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    }
    bag = BaggingClassifier(random_state=42)
    grid_bag = GridSearchCV(bag, bag_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_bag.fit(X_train, y_train)
    bag_best_params = grid_bag.best_params_

    # Evaluate Bagging model
    bag_accuracy, bag_f1, bag_recall, bag_precision = evaluate_model(grid_bag, X_test, y_test)
    results.append(['Bagging', bag_best_params, bag_accuracy, bag_f1, bag_recall, bag_precision])

    print("Finished Bagging")

    # 14. Quadratic Discriminant Analysis (QDA)
    qda_param_grid = {
        'reg_param': [0.0, 0.1, 0.5],
        'tol': [1e-4, 1e-3, 1e-2]
    }
    qda = QuadraticDiscriminantAnalysis()
    grid_qda = GridSearchCV(qda, qda_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_qda.fit(X_train, y_train)
    qda_best_params = grid_qda.best_params_

    # Evaluate QDA model
    qda_accuracy, qda_f1, qda_recall, qda_precision = evaluate_model(grid_qda, X_test, y_test)
    results.append(['QDA', qda_best_params, qda_accuracy, qda_f1, qda_recall, qda_precision])

    print("Finished QDA")

    # 15. Linear Discriminant Analysis (LDA)
    lda_param_grid = {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.5]
    }
    lda = LinearDiscriminantAnalysis()
    grid_lda = GridSearchCV(lda, lda_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_lda.fit(X_train, y_train)
    lda_best_params = grid_lda.best_params_

    # Evaluate LDA model
    lda_accuracy, lda_f1, lda_recall, lda_precision = evaluate_model(grid_lda, X_test, y_test)
    results.append(['LDA', lda_best_params, lda_accuracy, lda_f1, lda_recall, lda_precision])

    print("Finished LDA")

    # 16. Passive Aggressive Classifier
    pa_param_grid = {
        'C': [0.1, 1.0, 10],
        'max_iter': [1000, 2000],
        'tol': [1e-4, 1e-3]
    }
    pa = PassiveAggressiveClassifier(random_state=42)
    grid_pa = GridSearchCV(pa, pa_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_pa.fit(X_train, y_train)
    pa_best_params = grid_pa.best_params_

    # Evaluate PA model
    pa_accuracy, pa_f1, pa_recall, pa_precision = evaluate_model(grid_pa, X_test, y_test)
    results.append(['Passive Aggressive', pa_best_params, pa_accuracy, pa_f1, pa_recall, pa_precision])

    print("Finished Passive Aggressive")

    # 17. Perceptron
    perceptron_param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000],
        'tol': [1e-4, 1e-3]
    }
    perceptron = Perceptron(random_state=42)
    grid_perceptron = GridSearchCV(perceptron, perceptron_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_perceptron.fit(X_train, y_train)
    perceptron_best_params = grid_perceptron.best_params_

    # Evaluate Perceptron model
    perceptron_accuracy, perceptron_f1, perceptron_recall, perceptron_precision = evaluate_model(grid_perceptron, X_test, y_test)
    results.append(['Perceptron', perceptron_best_params, perceptron_accuracy, perceptron_f1, perceptron_recall, perceptron_precision])

    print("Finished Perceptron")

    # 18. Ridge Classifier
    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
    }
    ridge = RidgeClassifier()
    grid_ridge = GridSearchCV(ridge, ridge_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_ridge.fit(X_train, y_train)
    ridge_best_params = grid_ridge.best_params_

    # Evaluate Ridge Classifier
    ridge_accuracy, ridge_f1, ridge_recall, ridge_precision = evaluate_model(grid_ridge, X_test, y_test)
    results.append(['Ridge Classifier', ridge_best_params, ridge_accuracy, ridge_f1, ridge_recall, ridge_precision])

    print("Finished Ridge Classifier")


    # 19. SGD Classifier
    sgd_param_grid = {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000],
        'tol': [1e-4, 1e-3]
    }
    sgd = SGDClassifier(random_state=42)
    grid_sgd = GridSearchCV(sgd, sgd_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_sgd.fit(X_train, y_train)
    sgd_best_params = grid_sgd.best_params_

    # Evaluate SGD model
    sgd_accuracy, sgd_f1, sgd_recall, sgd_precision = evaluate_model(grid_sgd, X_test, y_test)
    results.append(['SGD Classifier', sgd_best_params, sgd_accuracy, sgd_f1, sgd_recall, sgd_precision])

    print("Finished SGD Classifier")

    # 20. Bagging Classifier with Decision Trees
    bagging_param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False]
    }
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)
    grid_bagging = GridSearchCV(bagging, bagging_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_bagging.fit(X_train, y_train)
    bagging_best_params = grid_bagging.best_params_

    # Evaluate Bagging model
    bagging_accuracy, bagging_f1, bagging_recall, bagging_precision = evaluate_model(grid_bagging, X_test, y_test)
    results.append(['Bagging Classifier with Decision Trees', bagging_best_params, bagging_accuracy, bagging_f1, bagging_recall, bagging_precision])

    print("Finished Bagging Classifier")


    # 21. Voting Classifier (Ensemble Method)
    voting_param_grid = {
        'voting': ['hard', 'soft']
    }
    voting = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('rf', RandomForestClassifier()), ('lr', LogisticRegression())])
    grid_voting = GridSearchCV(voting, voting_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_voting.fit(X_train, y_train)
    voting_best_params = grid_voting.best_params_

    # Evaluate Voting Classifier
    voting_accuracy, voting_f1, voting_recall, voting_precision = evaluate_model(grid_voting, X_test, y_test)
    results.append(['Voting Classifier', voting_best_params, voting_accuracy, voting_f1, voting_recall, voting_precision])

    print("Finished Voting Classifier")

    # 22. Logistic Regression (L1 Regularization)
    logistic_param_grid = {
        'penalty': ['l1'],  # Lasso equivalent
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear'],  # 'liblinear' supports L1 penalty
        'max_iter': [100, 200]
    }
    logistic = LogisticRegression(random_state=42)
    grid_logistic = GridSearchCV(logistic, logistic_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_logistic.fit(X_train, y_train)

    # Evaluate Logistic Regression model
    logistic_accuracy, logistic_f1, logistic_recall, logistic_precision = evaluate_model(grid_logistic, X_test, y_test)
    results.append(['Logistic Regression (L1)', logistic_param_grid, logistic_accuracy, logistic_f1, logistic_recall, logistic_precision])

    print("Finished Logistic Regression with L1 Regularization")


    lasso_param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000, 2000],
        'tol': [1e-4, 1e-3]
    }
    lasso = Lasso(random_state=42)
    grid_lasso = GridSearchCV(lasso, lasso_param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_lasso.fit(X_train, y_train)
    lasso_best_params = grid_lasso.best_params_
    # Predict continuous values
    y_pred_continuous = grid_lasso.predict(X_test)

    y_pred_discrete = np.where(y_pred_continuous < 0.5, 0, 1).astype('int')

    # Now you can evaluate using classification metrics
    lasso_accuracy = accuracy_score(y_test, y_pred_discrete)
    lasso_f1 = f1_score(y_test, y_pred_discrete)
    lasso_recall = recall_score(y_test, y_pred_discrete)
    lasso_precision = precision_score(y_test, y_pred_discrete)

    results.append(['Lasso Regression (Converted to Classification)', lasso_accuracy, lasso_f1, lasso_recall, lasso_precision])

    print("Finished Lasso (Converted to Classification)")
    # 23. Linear Regression
    linear_param_grid = {}  # Linear regression has no hyperparameters to tune by default
    linear = LinearRegression()
    grid_linear = GridSearchCV(linear, linear_param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_linear.fit(X_train, y_train)

    linear_best_params = grid_linear.best_params_
    # Predict continuous values
    y_pred_continuous = grid_linear.predict(X_test)
    y_pred_discrete = np.where(y_pred_continuous < 0.5, 0, 1).astype('int')
    # Evaluate Linear Regression model
    linear_accuracy = accuracy_score(y_test, y_pred_discrete)
    linear_f1 = f1_score(y_test, y_pred_discrete)
    linear_recall = recall_score(y_test, y_pred_discrete)
    linear_precision = precision_score(y_test, y_pred_discrete)
    results.append(['Linear Regression', None, linear_accuracy, linear_f1, linear_recall, linear_precision])

    print("Finished Linear Regression")



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

    # dl_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)


    # Evaluate DL model
    y_pred = dl_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    print(y_pred)

    print(y_test)


    dl_accuracy = accuracy_score(y_test, y_pred)
    dl_f1 = f1_score(y_test, y_pred)
    dl_recall = recall_score(y_test, y_pred)
    dl_precision = precision_score(y_test, y_pred)
    # results.append(['Deep Learning', {f"'epochs': {epochs}, 'batch_size': {batch_size}"}, dl_accuracy, dl_f1, dl_recall, dl_precision])

    print("Finished DL")

    # Export results to CSV
    results_df = pd.DataFrame(results, columns=['Model', 'Best Params', 'Accuracy', 'F1-Score', 'Recall', 'Precision'])
    results_df.to_csv(results_name, index=False)

    print(f"Results saved to '{results_name}'")
