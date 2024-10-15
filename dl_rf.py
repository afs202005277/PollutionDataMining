import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dl
import dl_test


def get_rf(X_train, y_train):
    # Random Forest configuration
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=9,
        min_samples_split=5,
        random_state=42
    )

    rf.fit(X_train, y_train)

    return rf


def main(name, data):
    hasHeadache_col = data.pop('hasHeadache')
    data['hasHeadache'] = hasHeadache_col
    # Split the dataset
    X = data.drop(columns=['hasHeadache'])
    y = data['hasHeadache']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_model = get_rf(X_train, y_train)

    rf_predictions = rf_model.predict(X_train)
    X_train = dl_test.standard_dataset(X_train)
    X_test = dl_test.standard_dataset(X_test)
    # 5. Append RF predictions as a new feature to the dataset
    X_train['RF_Predictions'] = rf_predictions
    rf_test_predictions = rf_model.predict(X_test)
    X_train['hasHeadache'] = y_train.values

    X_test['RF_Predictions'] = rf_test_predictions
    X_test['hasHeadache'] = y_test.values
    dl.create_model(name, X_train, X_test, 64)


if __name__ == "__main__":
    data = pd.read_csv('datasets/refined_embeddings/merged_data_processed.csv')
    main("augmented", data)