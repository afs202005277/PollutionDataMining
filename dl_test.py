import dl
import test
import pandas as pd
from sklearn.preprocessing import StandardScaler

def standard_dataset(df):
    # Step 1: Separate features and target variable
    X = df.drop(columns=['hasHeadache'])  # Exclude the target column
    y = df['hasHeadache']  # Target variable
    # Step 2: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Convert back to DataFrame (optional)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Step 4: Combine the scaled features with the target variable
    final_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    return final_df


paths = ['datasets/full_embeddings/', 'datasets/refined_embeddings/', 'datasets/pca/', 'datasets/initial/']
dl_names = ['fullembeddings_', 'refinedembeddings_', 'pca_', 'initial_']
names = ['data_processed.csv', 'data_processed_scaled.csv', 'merged_data_processed.csv', 'merged_data_processed_scaled.csv']
dll_names = ['normal', 'normal_scaled', 'augmented', 'augmented_scaled']

path = 'datasets/initial/'


for i, path in enumerate(paths):
    for j, name in enumerate(names):
        data = pd.read_csv(path + name)

        print('---------------------------------------------------')
        print("Random Forest:")
        test.runRF(dl_names[i] + dll_names[j], data)
        print('---------------------------------------------------')
        print('Decision Tree:')
        test.runDT(dl_names[i] + dll_names[j], data)
        print('---------------------------------------------------')
        '''divide_point = int(0.7*len(data))

        data_train = data[:divide_point]
        data_test = data[divide_point:]
        print('---------------------------------------------------')

        dl.create_model(dl_names[i] + dll_names[j], data_train, data_test, 64)
        
        print('Model ' + dl_names[i] + dll_names[j] + ' created')
        print('---------------------------------------------------')'''

