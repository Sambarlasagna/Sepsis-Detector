import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Sepsis Data Preprocessing and Sequence Preparation

This script processes the raw sepsis dataset to prepare it for
time-to-event modeling (predicting time until sepsis onset).

Key steps performed:

1. Data Exploration
   - Displays basic dataset info and a preview of the data.

2. Data Preprocessing
   - Drops redundant columns.
   - Sorts data by patient ID and hourly time index to preserve temporal order.
   - Handles missing values by forward/backward filling per patient,
     then fills any remaining missing data with median values.
   - Normalizes features using z-score scaling.

3. Label Engineering: Time-to-Sepsis
   - Creates a new label column 'time_to_sepsis' that indicates,
     for each hourly record, how many hours remain until sepsis onset.
   - For patients without sepsis, labels are capped at a max value (censored).

4. Sequence Creation for Modeling
   - Generates fixed-length sliding window sequences of patient data.
   - Each sequence consists of consecutive hourly feature values.
   - Labels correspond to 'time_to_sepsis' at the end of each sequence.

5. Output
   - Processed DataFrame is saved as 'processed_sepsis_dataset.csv'.
   - Prepared sequences (X) and labels (y) are ready for use in predictive models.

This enables training machine learning models (e.g., LSTM/GRU) to predict
how soon sepsis will occur given patient data sequences in real-time.
"""


def preprocess_sepsis_data(df):
    """
    Preprocess the raw sepsis dataset:
    - Drops redundant columns
    - Sorts data by Patient_ID and Hour to maintain temporal order
    - Handles missing data by forward and backward filling per patient
    - Fills any remaining NaNs with the median of the respective feature
    - Applies z-score normalization to features
    Returns:
        df: cleaned and normalized dataframe
        features: list of feature column names used for modeling
    """
    # Drop redundant column if present
    df = df.drop(columns=['Unnamed: 0'])

    # Sort data by patient and hourly time index
    df = df.sort_values(['Patient_ID', 'Hour'])

    # Identify features excluding target and identifiers
    features = [col for col in df.columns if col not in ['SepsisLabel', 'Patient_ID', 'Hour']]
    
    # Forward and backward fill missing data per patient to keep continuity
    df[features] = df.groupby('Patient_ID')[features].transform(lambda group: group.ffill().bfill())

    # Fill any remaining missing data with median values (robust to outliers)
    df[features] = df[features].fillna(df[features].median())

    # Standardize feature values to zero mean and unit variance
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    print("Preprocessing complete. Sample data:")
    print(df.head())

    return df, features


def explore_csv(df):
    """
    Exploratory function to inspect dataset structure and contents.
    Prints column names, datatype info, and first few rows for overview.
    """
    print("Columns in sepsis_dataset.csv:")
    print(df.columns.tolist())
    print("\nDataframe Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())


def create_time_to_sepsis_label(df, patient_col='Patient_ID', hour_col='Hour', label_col='SepsisLabel', max_time=6):
    """
    Create a time-to-event label column 'time_to_sepsis':
    - For each hour in patient data, this column indicates how many hours remain until sepsis onset.
    - Hours after or at sepsis onset have label 0.
    - Patients without sepsis have label set to max_time (censored).
    Args:
        df: input preprocessed dataframe
        patient_col: name of patient identifier column
        hour_col: name of time index column (usually hours)
        label_col: binary sepsis onset label column
        max_time: maximum cap for time-to-sepsis (censoring value)
    Returns:
        df: dataframe with added 'time_to_sepsis' column
    """
    df = df.copy()
    df['time_to_sepsis'] = max_time  # Initialize with maximum time (censored)

    patients = df[patient_col].unique()

    for patient in patients:
        patient_data = df[df[patient_col] == patient]
        sepsis_hours = patient_data[patient_data[label_col] == 1][hour_col].values

        # If no sepsis, keep max_time as label
        if len(sepsis_hours) == 0:
            continue

        sepsis_start = sepsis_hours[0]  # First hour of sepsis onset

        # Calculate time difference to sepsis at each hour, clipped between 0 and max_time
        time_to_sepsis_vals = (sepsis_start - patient_data[hour_col]).clip(lower=0, upper=max_time).values

        # Update dataframe with calculated time-to-sepsis for this patient
        df.loc[patient_data.index, 'time_to_sepsis'] = time_to_sepsis_vals

    return df


def create_sequences(df, patient_col='Patient_ID', hour_col='Hour', feature_cols=None, seq_length=6):
    """
    Generate fixed-length sequences of patient data for model input:
    - For each patient, slide a window of length seq_length over their hourly data.
    - Each sequence contains features for consecutive hours.
    - The label for a sequence is the 'time_to_sepsis' value at the last hour of that sequence.
    Args:
        df: input dataframe with preprocessed features and 'time_to_sepsis' label
        patient_col: patient identifier column
        hour_col: hourly time column
        feature_cols: list of feature columns to include in sequences (if None, all except IDs and labels are used)
        seq_length: length of each input sequence (e.g. 12 for 12 hours)
    Returns:
        X: numpy array of shape (num_sequences, seq_length, num_features) with input sequences
        y: numpy array of shape (num_sequences,) with corresponding time-to-sepsis labels
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [patient_col, hour_col, 'SepsisLabel', 'time_to_sepsis']]

    sequences = []
    labels = []

    patients = df[patient_col].unique()

    for patient in patients:
        patient_data = df[df[patient_col] == patient].sort_values(hour_col)

        features_array = patient_data[feature_cols].values
        time_to_sepsis_array = patient_data['time_to_sepsis'].values

        for start in range(len(patient_data) - seq_length + 1):
            end = start + seq_length
            seq_X = features_array[start:end]
            seq_y = time_to_sepsis_array[end - 1]  # label at last hour of sequence

            sequences.append(seq_X)
            labels.append(seq_y)

    X = np.array(sequences)
    y = np.array(labels)

    print(f"Created {len(X)} sequences each of length {seq_length}.")

    return X, y

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sepsis_dataset.csv')
    df = pd.read_csv(csv_path)
    # Explore the CSV file
    explore_csv(df)
    # Load data
    df,features = preprocess_sepsis_data(df)  # your cleaned dataframe sorted by Patient_ID and Hour
    df = create_time_to_sepsis_label(df)
    # Now df['time_to_sepsis'] contains targets for time-to-sepsis prediction
    # X, y = create_sequences(df, feature_cols=features, seq_length=12)
    # np.save(os.path.join(os.path.dirname(__file__), '..', 'data', 'X_sequences.npy'), X)
    # np.save(os.path.join(os.path.dirname(__file__), '..', 'data', 'y_labels.npy'), y)
    # print("Saved X and y sequences as numpy files.")

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv')
    df.to_csv(output_path, index=False)
    print("Processed data saved to 'processed_sepsis_dataset.csv'")


