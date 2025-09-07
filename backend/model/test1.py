
import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv')
csv_path2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'patient1_sepsis_simulated.csv')
df = pd.read_csv(csv_path)
print(df.head(20))  # prints first 20 rows to see new columns and values

df2 = pd.read_csv(csv_path2)
print(df2.head(20))  # prints first 20 rows to see new columns and values


print(df['time_to_sepsis'].value_counts())  # check distribution of new label

