from unittest.mock import inplace

import pandas as pd
import numpy as np

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

def prepare_data():
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

    allergies = ['penicillin', 'cat', 'dog', 'dust', 'ragweed', 'morphine', 'shrimp', 'shellfish',
                 'anaphylaxis', 'iodine', 'pcn', 'ceclor', 'sulfa', 'latex', 'peanut', 'mold', 'bee', 'nut',
                 'aspirin', 'pepcid', 'doxycicline', 'crestor', 'cockroach', 'codeine', 'inh', 'seafood',
                 'mushroom', 'chlorthalidone', 'grass', 'amoxicillin', 'fish',
                 'seasonal', 'percocet', 'acetaminophen', 'lovonox', 'asa', 'motrin', 'bugs', 'ceftriaxone', 'ees',
                 'wheat', 'gluten', 'vancomycin', 'msg', 'dilaudid', 'nsais', 'naproxen', 'benadryl',
                 'dairy', 'ibuprofen', 'kiwi', 'bactrim', 'erytromycin', 'benzoyl', 'keflex', 'lidocaine',
                 'pineapple', 'clindamycin', 'indocin', 'demerol', 'lorabid', 'sulfur', 'sulfer', 'strawberry',
                 'vicodin']

    for allergy in allergies:
        data[f'has_{allergy}_allergy'] = data['ALLERGIES'].str.lower().str.contains(allergy, case=False, na=False)

    meds = ['acetaminophen', 'ibuprofen', 'aspirin', 'diphenhydramine', 'paracetamol',
            'insulin', 'benzoyl', 'synthroid', 'fluvoxamine', 'singulair', 'ozempic', 'avapro',
            'zyrtec', 'propanol', 'prozac', 'loratadine', 'metformin', 'lisinopril', 'metoprolol',
            'losartan', 'albuterol', 'levothyroxine', 'omeprazole', 'simvastatin', 'hydrochlorothiazide',
            'lasix', 'celecoxib', 'flurbiprofen', 'fenoprofen']

    for med in meds:
        data[f'use_{med}'] = data['OTHER_MEDS'].str.lower().str.contains(allergy, case=False, na=False)

    data['has_migraine'] = data['HISTORY'].str.lower().str.contains('migraine', case=False, na=False)

    cols = ['ALLERGIES', 'OTHER_MEDS', 'HISTORY']

    data = data.drop(cols, axis=1)

    n = len(data[data['hasHeadache'] == 1])

    data = pd.concat([data[data['hasHeadache'] == 1], data[data['hasHeadache'] == 0].sample(n)])

    data = data.sample(frac=1)

    pollution_data = pollution_data.groupby('State')[['AirQuality', 'WaterPollution']].mean().reset_index()

    merged_df = pd.merge(data, pollution_data, on='State', suffixes=('_df1', '_df2'))

    merged_df.to_csv('data_processed.csv', index=False)


if __name__ == '__main__':
    prepare_data()
