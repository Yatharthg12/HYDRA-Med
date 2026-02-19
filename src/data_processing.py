import pandas as pd
import numpy as np


def load_and_clean_data(filepath="data/diabetic_data_reduced.csv", sample_size=None):
    df = pd.read_csv(filepath)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    df['readmitted_binary'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    df = df.drop(['encounter_id', 'patient_nbr'], axis=1)
    df.replace('?', np.nan, inplace=True)

    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > 0.4].index
    df = df.drop(columns=cols_to_drop)

    df = df.drop(columns=['readmitted'])

    return df


def prepare_features(df):
    X = df.drop(columns=['readmitted_binary'])
    y = df['readmitted_binary']

    X = pd.get_dummies(X, drop_first=True)

    return X, y
