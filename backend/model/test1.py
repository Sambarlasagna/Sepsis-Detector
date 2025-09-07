import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_patient_dataset.csv')

# Load the CSV
df = pd.read_csv(DATA_PATH)

# Show basic info
print("Columns:", df.columns.tolist())
print("\nFirst 10 rows:\n", df.head(10))
print("\nData summary:\n", df.describe())

# Optional: see per-hour trends for the first patient
patient_id = df['Patient_ID'].unique()[0]
patient_df = df[df['Patient_ID'] == patient_id]
print("\nPatient-specific data (all hours):\n", patient_df)
