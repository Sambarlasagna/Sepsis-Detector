import pandas as pd
import numpy as np

# 24 hours range
hours = np.arange(0, 25)

# Initialize dataframe
patient1_df = pd.DataFrame({'Hour': hours})

# Simulate rapid progression for sepsis by hour 12 for Patient_ID 1
# Faster escalation within 12 hours, then plateau or persist till 24 hours
def accelerated_linspace(start, end, steps, midpoint=12, after_val=None):
    """Creates a linspace that reaches 'end' at 'midpoint' then stays constant"""
    vals = np.concatenate([
        np.linspace(start, end, midpoint + 1),
        np.full(steps - (midpoint + 1), end if after_val is None else after_val)
    ])
    return vals

patient1_df['HR'] = np.clip(accelerated_linspace(75, 130, 25) + np.random.normal(0, 3, 25), 65, 145)
patient1_df['O2Sat'] = np.clip(accelerated_linspace(97, 85, 25) + np.random.normal(0, 0.7, 25), 80, 100)
patient1_df['Temp'] = np.clip(accelerated_linspace(36.7, 40.2, 25) + np.random.normal(0, 0.2, 25), 35.8, 41)
patient1_df['SBP'] = np.clip(accelerated_linspace(125, 75, 25) + np.random.normal(0, 2.5, 25), 65, 130)
patient1_df['MAP'] = np.clip(accelerated_linspace(88, 58, 25) + np.random.normal(0, 2, 25), 50, 100)
patient1_df['DBP'] = np.clip(accelerated_linspace(79, 53, 25) + np.random.normal(0, 2, 25), 35, 90)
patient1_df['Resp'] = np.clip(accelerated_linspace(14, 34, 25) + np.random.normal(0, 1.1, 25), 10, 40)
patient1_df['EtCO2'] = np.clip(accelerated_linspace(36, 20, 25) + np.random.normal(0, 2, 25), 12, 45)
patient1_df['BaseExcess'] = np.clip(accelerated_linspace(1, -7, 25) + np.random.normal(0, 0.5, 25), -10, 3)
patient1_df['HCO3'] = np.clip(accelerated_linspace(25, 15, 25) + np.random.normal(0, 0.5, 25), 10, 26)
patient1_df['FiO2'] = np.clip(np.linspace(0.21, 0.6, 25), 0.21, 1)  # oxygen support rising gradually
patient1_df['pH'] = np.clip(accelerated_linspace(7.4, 7.18, 25) + np.random.normal(0, 0.02, 25), 7.0, 7.45)
patient1_df['PaCO2'] = np.clip(accelerated_linspace(38, 52, 25) + np.random.normal(0, 3, 25), 30, 70)
patient1_df['SaO2'] = np.clip(accelerated_linspace(98, 85, 25) + np.random.normal(0, 1, 25), 80, 100)
patient1_df['AST'] = np.clip(accelerated_linspace(15, 110, 25) + np.random.normal(0, 12, 25), 10, 150)
patient1_df['BUN'] = np.clip(accelerated_linspace(9, 50, 25) + np.random.normal(0, 4, 25), 5, 65)
patient1_df['Alkalinephos'] = np.clip(accelerated_linspace(65, 160, 25) + np.random.normal(0, 15, 25), 40, 200)
patient1_df['Calcium'] = np.clip(accelerated_linspace(9.1, 7.2, 25) + np.random.normal(0, 0.25, 25), 6.5, 10)
patient1_df['Chloride'] = np.clip(accelerated_linspace(98, 114, 25) + np.random.normal(0, 3, 25), 90, 120)
patient1_df['Creatinine'] = np.clip(accelerated_linspace(0.95, 3.0, 25) + np.random.normal(0, 0.3, 25), 0.7, 4)
patient1_df['Bilirubin_direct'] = np.clip(accelerated_linspace(0.05, 1.4, 25) + np.random.normal(0, 0.15, 25), 0.0, 3)
patient1_df['Glucose'] = np.clip(accelerated_linspace(85, 170, 25) + np.random.normal(0, 10, 25), 70, 220)
patient1_df['Lactate'] = np.clip(accelerated_linspace(0.8, 7.5, 25) + np.random.normal(0, 0.5, 25), 0.5, 12)
patient1_df['Magnesium'] = np.clip(accelerated_linspace(1.7, 2.1, 25) + np.random.normal(0, 0.12, 25), 1.2, 3)
patient1_df['Phosphate'] = np.clip(accelerated_linspace(3.1, 4.9, 25) + np.random.normal(0, 0.25, 25), 2.5, 6)
patient1_df['Potassium'] = np.clip(accelerated_linspace(3.8, 6.0, 25) + np.random.normal(0, 0.3, 25), 3.5, 7.2)
patient1_df['Bilirubin_total'] = np.clip(accelerated_linspace(0.6, 2.8, 25) + np.random.normal(0, 0.3, 25), 0.5, 4.5)
patient1_df['TroponinI'] = np.clip(accelerated_linspace(0, 0.4, 25) + np.random.normal(0, 0.03, 25), 0, 1.0)
patient1_df['Hct'] = np.clip(accelerated_linspace(42, 33, 25) + np.random.normal(0, 1.5, 25), 28, 45)
patient1_df['Hgb'] = np.clip(accelerated_linspace(14, 10.5, 25) + np.random.normal(0, 0.4, 25), 8, 15)
patient1_df['PTT'] = np.clip(accelerated_linspace(29, 54, 25) + np.random.normal(0, 3, 25), 20, 70)
patient1_df['WBC'] = np.clip(accelerated_linspace(7, 22, 25) + np.random.normal(0, 2, 25), 4, 25)
patient1_df['Fibrinogen'] = np.clip(accelerated_linspace(280, 515, 25) + np.random.normal(0, 25, 25), 200, 600)
patient1_df['Platelets'] = np.clip(accelerated_linspace(265, 135, 25) + np.random.normal(0, 12, 25), 100, 350)

patient1_df['Age'] = 60
patient1_df['Gender'] = 1
patient1_df['Unit1'] = 1
patient1_df['Unit2'] = 0
patient1_df['HospAdmTime'] = -90
patient1_df['ICULOS'] = patient1_df['Hour']

# Sepsis label 1 from hour 12 onward
patient1_df['SepsisLabel'] = (patient1_df['Hour'] >= 12).astype(int)
patient1_df['Patient_ID'] = 1
patient1_df['time_to_sepsis'] = np.where(patient1_df['Hour'] <= 12, 12 - patient1_df['Hour'], 0)

# Round values
cols_to_round = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
                 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

for col in cols_to_round:
    patient1_df[col] = patient1_df[col].round(2)

# Display first rows
print(patient1_df.head(15))
patient1_df.to_csv('patient1_sepsis_simulated.csv', index=False)