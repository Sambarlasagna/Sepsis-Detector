import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_sepsis_data(csv_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Drop redundant columns
    df = df.drop(columns=['Unnamed: 0'])

    # Sort by Patient_ID and Hour to maintain sequence order
    df = df.sort_values(['Patient_ID', 'Hour'])

    # List of features excluding target and identifiers
    features = [col for col in df.columns if col not in ['SepsisLabel', 'Patient_ID', 'Hour']]
    # Handle missing data: forward fill and backward fill per patient using transform
    df[features] = df.groupby('Patient_ID')[features].transform(lambda group: group.ffill().bfill())


    # After filling, some columns may still have NaNs, fill remaining with median
    df[features] = df[features].fillna(df[features].median())

    # Feature scaling - z-score normalization per feature
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # The dataset is now ready for model input
    # You might want to convert to sequences grouped by Patient_ID for models like LSTM

    print("Preprocessing complete. Sample data:")
    print(df.head())

    return df, features

def explore_csv():
    # Construct the relative path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sepsis_dataset.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Display basic info and columns
    print("Columns in sepsis_dataset.csv:")
    print(df.columns.tolist())
    print("\nDataframe Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == '__main__':
    # Explore the CSV file
    explore_csv()
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sepsis_dataset.csv')
    preprocess_sepsis_data(csv_path)
