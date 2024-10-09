import pandas as pd
    
data1 = pd.read_csv('data_processed.csv')
data2 = pd.read_csv('data_processed_scaled.csv')
data3 = pd.read_csv('merged_data_processed.csv')
data4 = pd.read_csv('merged_data_processed_scaled.csv')

data1.drop('hadHeadIllness', axis=1, inplace=True)
data2.drop('hadHeadIllness', axis=1, inplace=True)
data3.drop('hadHeadIllness', axis=1, inplace=True)
data4.drop('hadHeadIllness', axis=1, inplace=True)

data1.to_csv('data_processed.csv', index=False)
data2.to_csv('data_processed_scaled.csv', index=False)
data3.to_csv('merged_data_processed.csv', index=False)
data4.to_csv('merged_data_processed_scaled.csv', index=False)