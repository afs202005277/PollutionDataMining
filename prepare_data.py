import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

abbreviation_to_state = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

state_to_abbreviation = {state: abbrev for abbrev, state in abbreviation_to_state.items()}

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

def get_batch_embeddings(texts, batch_size=64):
    embeddings = []
    first_time = True
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        # Average the embeddings for the entire batch
        embeddings.extend(outputs.last_hidden_state.mean(dim=1).numpy())

        if first_time:
            duration = time.time() - start
            print(duration)
            print(f"Estimated time: {round(len(texts)/ batch_size *  duration / 60 / 60, 2)} hours.")
            first_time = False
    return embeddings

def scree_plot(raw_embed):
    # Assume raw_embed is your embeddings array
    pca = PCA()  # Fit PCA without specifying n_components
    pca.fit(raw_embed)

    # Plot the cumulative explained variance
    #plt.figure(figsize=(8, 6))
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.xlabel('Number of Components')
    #plt.ylabel('Cumulative Explained Variance')
    #plt.title('Explained Variance vs. Number of Components')
    #plt.grid(True)
    #plt.show()

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1  # 95% threshold
    print(f"Number of components that explain 95% of variance: {n_components}")
    return n_components

def prepare_data():
    columns_to_embeddings = ['HISTORY', 'ALLERGIES']
    data1 = pd.read_csv('data/VAERSDATA.csv')
    data2 = pd.read_csv('data/VAERSVAX.csv')
    data3 = pd.read_csv('data/VAERSSYMPTOMS.csv')

    pollution_data = pd.read_csv('data/water_air_pollution.csv')

    pollution_data.columns = pollution_data.columns.str.replace(r"[\"']", "", regex=True).str.strip()
    pollution_data = pollution_data.applymap(lambda x: x.replace('"', '').replace("'", "").strip() if isinstance(x, str) else x)

    pollution_data = pollution_data[pollution_data['Country'] == 'United States of America']

    # merge data based on VAERS_ID
    data = pd.merge(data1, data2, on='VAERS_ID')
    data = pd.merge(data, data3, on='VAERS_ID')
    
    # data['hadHeadIllness'] = np.where((data['CUR_ILL'].str.contains('head|Head', na=False)) | (data['SYMPTOM_TEXT'].str.contains('head|Head', na=False)), 1, 0)

    # Check if some of the symptoms columns have a 'headache' as value, if yes, create a new column with a boolean value

    data['hasHeadache'] = np.where(
        (data['SYMPTOM1'] == 'Headache') | (data['SYMPTOM2'] == 'Headache') | (data['SYMPTOM3'] == 'Headache') | (
                data['SYMPTOM4'] == 'Headache') | (data['SYMPTOM5'] == 'Headache'), 1, 0)

    cols = ['VAERS_ID', 'CAGE_YR', 'RECVDATE', 'TODAYS_DATE', 'SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4',
            'SYMPTOM5',
            'SYMPTOMVERSION1', 'SYMPTOMVERSION2', 'SYMPTOMVERSION3', 'SYMPTOMVERSION4', 'SYMPTOMVERSION5',
            'SYMPTOM_TEXT', 'LAB_DATA', 'CUR_ILL']

    data = data.drop(cols, axis=1)

    missing_percentage = data.isnull().mean() * 100
    data = data.loc[:, missing_percentage <= 60]

    data['VAX_DATE'] = (pd.to_datetime(data['VAX_DATE']).astype('int64') // 10 ** 9).astype('int32')
    data['ONSET_DATE'] = (pd.to_datetime(data['ONSET_DATE']).astype('int64') // 10 ** 9).astype('int32')
    # data['TODAYS_DATE'] = (pd.to_datetime(data['TODAYS_DATE']).astype('int64') // 10 ** 9).astype('int32')

    data['RECOVD'].replace({'N': -1, 'U': 0, 'Y': 1}, inplace=True)
    data['RECOVD'].fillna(0, inplace=True)

    data['VAX_DOSE_SERIES'].replace({'7+': 7.5}, inplace=True)

    data['ALLERGIES'] = data['ALLERGIES'].replace({'(?i)no': np.nan, '(?i)none': np.nan, '(?i)na': np.nan, '(?i)zero': np.nan, '(?i)n/a': np.nan, '(?i)unknown': np.nan, '(?i)unk': np.nan, '0': np.nan, '(?i)n.a': np.nan}, regex=True)
    # data['ALLERGIES'] = data['ALLERGIES'].notna() & (data['ALLERGIES'] != '')

    data['OTHER_MEDS'] = data['OTHER_MEDS'].replace({'(?i)no': np.nan, '(?i)none': np.nan, '(?i)na': np.nan, '(?i)zero': np.nan, '(?i)n/a': np.nan, '(?i)unknown': np.nan, '(?i)unk': np.nan, '0': np.nan, '(?i)n.a': np.nan}, regex=True)
    data['OTHER_MEDS'] = data['OTHER_MEDS'].notna() & (data['OTHER_MEDS'] != '')

    data['VAX_LOT'].replace(' ', '', inplace=True)
    data['VAX_LOT'] = pd.factorize(data['VAX_LOT'])[0]
    data['VAX_LOT'].fillna(0, inplace=True)

    data['VAX_SITE'] = pd.factorize(data['VAX_SITE'])[0]
    data['VAX_SITE'].fillna(-1, inplace=True)

    data['VAX_ROUTE'] = pd.factorize(data['VAX_ROUTE'])[0]
    data['VAX_ROUTE'].fillna(-1, inplace=True)

    data['SEX'] = pd.factorize(data['SEX'])[0]

    data['V_ADMINBY'] = pd.factorize(data['V_ADMINBY'])[0]

    data['VAX_TYPE'] = pd.factorize(data['VAX_TYPE'])[0]

    data['VAX_MANU'] = pd.factorize(data['VAX_MANU'])[0]

    data['VAX_NAME'] = pd.factorize(data['VAX_NAME'])[0]

    # data['CAGE_YR'].fillna(data['CAGE_YR'].mean(), inplace=True)

    data.rename(columns={'STATE': 'State'}, inplace=True)
    data = data.dropna(subset=['State'])
    pollution_data.rename(columns={'Region': 'State'}, inplace=True)

    pollution_data['State'] = pollution_data['State'].replace('District of Columbia', 'Washington')

    pollution_data['State'] = pollution_data['State'].map(state_to_abbreviation)

    state_labels, unique_states = pd.factorize(data['State'])
    # Step 2: Create a mapping dictionary from the unique states and their factorized labels
    state_mapping = {state: idx for idx, state in enumerate(unique_states)}

    # Step 3: Apply the same factorization to df2 using the mapping
    pollution_data['State'] = pollution_data['State'].map(state_mapping)
    data['State'] = data['State'].map(state_mapping)
    pollution_data['State'].fillna(-1, inplace=True)

    data['NUMDAYS'].fillna(data['NUMDAYS'].mode()[0], inplace=True)

    data['AGE_YRS'].fillna(data['AGE_YRS'].mean(), inplace=True)

    data['ONSET_DATE'].dropna(inplace=True)
    data['VAX_DATE'].dropna(inplace=True)
    # data['TODAYS_DATE'].dropna(inplace=True)

    data['VAX_DOSE_SERIES'].replace('UNK', 0, inplace=True)
    data['VAX_DOSE_SERIES'].fillna(0, inplace=True)

    data['has_migraine'] = data['HISTORY'].str.lower().str.contains('migraine', case=False, na=False)
    data['HISTORY'] = data['HISTORY'].replace(
        {'(?i)no': np.nan, '(?i)none': np.nan, '(?i)na': np.nan, '(?i)zero': np.nan, '(?i)n/a': np.nan,
         '(?i)unknown': np.nan, '(?i)unk': np.nan, '0': np.nan, '(?i)n.a': np.nan}, regex=True)

    for column_name in columns_to_embeddings:
        data[column_name].fillna("unknown", inplace=True)

        raw_embed = np.array(get_batch_embeddings(data[column_name].tolist()))

        #n_components = scree_plot(raw_embed)
        #pca = PCA(n_components=n_components, random_state=42)
        #reduced_embeddings = pca.fit_transform(raw_embed)
        embeddings = pd.DataFrame(raw_embed,
                                  columns=[f"{column_name}_dim_{i + 1}" for i in range(raw_embed.shape[1])])
        data = pd.concat([data.drop(column_name, axis=1), embeddings], axis=1)
        print(f"Finished concatenating {column_name}\n")

    n = len(data[data['hasHeadache'] == 1])

    data = pd.concat([data[data['hasHeadache'] == 1], data[data['hasHeadache'] == 0].sample(n)])

    data = data.sample(frac=1)

    pollution_data = pollution_data.groupby('State')[['AirQuality', 'WaterPollution']].mean().reset_index()

    data.dropna(inplace=True)
    merged_df = pd.merge(data, pollution_data, on='State', suffixes=('_df1', '_df2'))

    merged_df = merged_df.drop(columns=['State'])
    merged_df.columns = merged_df.columns.astype(str)
    data.columns = data.columns.astype(str)


    data[:200].to_csv('sample.csv', index=False)
    merged_df.to_csv('merged_data_processed.csv', index=False)
    data.to_csv('data_processed.csv', index=False)

    merged_df = standard_dataset(merged_df)
    data = standard_dataset(data)

    data[:200].to_csv('sample_scaled.csv', index=False)
    merged_df.to_csv('merged_data_processed_scaled.csv', index=False)
    data.to_csv('data_processed_scaled.csv', index=False)


if __name__ == '__main__':
    prepare_data()
