import pandas as pd
import os
# Load your dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv'))  # replace with actual path

# Find the index of the first row where SepsisLabel == 1
sepsis_index = df[df['SepsisLabel'] == 1].index[0]

# Calculate start index to get 6 rows before SepsisLabel == 1
start_index = max(sepsis_index - 6, 0)  # handle case if index < 6

# Slice dataframe to get 6 rows before and including SepsisLabel == 1 row
rows_to_view = df.iloc[start_index:sepsis_index + 1]
print(df["Hour"].max())

print(rows_to_view)
