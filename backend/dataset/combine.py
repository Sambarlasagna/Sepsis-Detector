import pandas as pd

# Load the CSV files
patient1_df = pd.read_csv('dataset/patient1_sepsis_simulated.csv')
patient3_df = pd.read_csv('dataset/patient3_sepsis_simulated.csv')
simulated_df = pd.read_csv('dataset/simulated_sepsis_data.csv')
# Combine datasets
combined_df = pd.concat([patient1_df, patient3_df, simulated_df], ignore_index=True)
# Optional: check combined shape and head
print(f'Combined dataset shape: {combined_df.shape}')
print(combined_df.head())

# Save combined dataset to CSV
combined_df.to_csv('combined_sepsis_dataset.csv', index=False)
print('Combined dataset saved as combined_sepsis_dataset.csv')