import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  # import joblib for saving scaler


SEQ_LENGTH = 6
MAX_TIME = 6


DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sepsis_dataset.csv')
PROCESSED_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv')
SCALER_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'scaler.save')  # path to save scaler


def preprocess_sepsis_data(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df.sort_values(['Patient_ID', 'Hour'])
    features = [c for c in df.columns if c not in ['SepsisLabel', 'Patient_ID', 'Hour']]
    
    df[features] = df.groupby('Patient_ID')[features].transform(lambda g: g.ffill().bfill())
    df[features] = df[features].fillna(df[features].median())
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save the fitted scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    return df, features


def create_time_to_sepsis_label(df, max_time=MAX_TIME):
    df = df.copy()
    df['time_to_sepsis'] = max_time
    for patient in df['Patient_ID'].unique():
        pdata = df[df['Patient_ID'] == patient]
        sepsis_hours = pdata[pdata['SepsisLabel'] == 1]['Hour'].values
        if len(sepsis_hours) == 0:
            continue
        onset = sepsis_hours[0]
        tts = (onset - pdata['Hour']).clip(0, max_time).values
        df.loc[pdata.index, 'time_to_sepsis'] = tts
    return df


def main():
    df_raw = pd.read_csv(DATA_CSV_PATH)
    df_proc, features = preprocess_sepsis_data(df_raw)
    df_proc = create_time_to_sepsis_label(df_proc)
    df_proc.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_CSV_PATH}")
    
if __name__ == "__main__":
    main()
