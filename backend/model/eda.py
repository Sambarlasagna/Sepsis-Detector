import pandas as pd

# Example: Load one patient file (replace with actual filepath)
file_path = 'path_to_patient_file.psv'
data = pd.read_csv(file_path, sep='|')

print(data.head())
print(data.columns)
print(data.isnull().sum())
