
import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv')
df = pd.read_csv(csv_path)
print(df.head(20))  # prints first 20 rows to see new columns and values
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))

