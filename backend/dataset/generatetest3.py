import pandas as pd
import numpy as np

# 24 hours range
hours = np.arange(0, 25)

# Initialize dataframe
patient3_df = pd.DataFrame({'Hour': hours})

def accelerated_linspace(start, end, steps, midpoint=6, after_val=None):
    """
    Creates a linspace that reaches 'end' at 'midpoint' then stays constant
    """
    vals = np.concatenate([
        np.linspace(start, end, midpoint + 1),
        np.full(steps - (midpoint + 1), end if after_val is None else after_val)
    ])
    return vals

# Simulate accelerated sepsis progression for Patient_ID 3 (sepsis by hour 6)
patient3_df['HR'] = np.clip(accelerated_linspace(70, 140, 25) + np.random.normal(0, 4, 25), 60, 150)
patient3_df['O2Sat'] = np.clip(accelerated_linspace(98, 82, 25) + np.random.normal(0, 1, 25), 75, 100)
patient3_df['Temp'] = np.clip(accelerated_linspace(36.5, 40.5, 25) + np.random.normal(0, 0.3, 25), 35, 42)
patient3_df['SBP'] = np.clip(accelerated_linspace(130, 70, 25) + np.random.normal(0, 3, 25), 60, 135)
patient3_df['MAP'] = np.clip(accelerated_linspace(90, 55, 25) + np.random.normal(0, 2.5, 25), 40, 100)
patient3_df['DBP'] = np.clip(accelerated_linspace(80, 50, 25) + np.random.normal(0, 2, 25), 30, 90)
patient3_df['Resp'] = np.clip(accelerated_linspace(16, 38, 25) + np.random.normal(0, 1.5, 25), 10, 45)
patient3_df['EtCO2'] = np.clip(accelerated_linspace(37, 18, 25) + np.random.normal(0, 3, 25), 10, 50)
patient3_df['BaseExcess'] = np.clip(accelerated_linspace(1, -8, 25) + np.random.normal(0, 0.7, 25), -12, 4)
patient3_df['HCO3'] = np.clip(accelerated_linspace(26, 14, 25) + np.random.normal(0, 0.7, 25), 8, 26)
patient3_df['FiO2'] = np.linspace(0.21, 0.7, 25)
patient3_df['pH'] = np.clip(accelerated_linspace(7.43, 7.12, 25) + np.random.normal(0, 0.03, 25), 6.9, 7.5)
patient3_df['PaCO2'] = np.clip(accelerated_linspace(37, 56, 25) + np.random.normal(0, 4, 25), 25, 80)
patient3_df['SaO2'] = np.clip(accelerated_linspace(99, 83, 25) + np.random.normal(0, 1.5, 25), 70, 100)
patient3_df['AST'] = np.clip(accelerated_linspace(14, 130, 25) + np.random.normal(0, 15, 25), 10, 180)
patient3_df['BUN'] = np.clip(accelerated_linspace(7, 60, 25) + np.random.normal(0, 5, 25), 5, 70)
patient3_df['Alkalinephos'] = np.clip(accelerated_linspace(65, 190, 25) + np.random.normal(0, 20, 25), 40, 210)
patient3_df['Calcium'] = np.clip(accelerated_linspace(9.4, 6.8, 25) + np.random.normal(0, 0.3, 25), 6, 10)
patient3_df['Chloride'] = np.clip(accelerated_linspace(99, 116, 25) + np.random.normal(0, 4, 25), 85, 125)
patient3_df['Creatinine'] = np.clip(accelerated_linspace(0.8, 3.5, 25) + np.random.normal(0, 0.4, 25), 0.6, 5)
patient3_df['Bilirubin_direct'] = np.clip(accelerated_linspace(0.04, 1.7, 25) + np.random.normal(0, 0.2, 25), 0, 3.5)
patient3_df['Glucose'] = np.clip(accelerated_linspace(80, 190, 25) + np.random.normal(0, 15, 25), 60, 250)
patient3_df['Lactate'] = np.clip(accelerated_linspace(1, 9, 25) + np.random.normal(0, 0.6, 25), 0.4, 15)
patient3_df['Magnesium'] = np.clip(accelerated_linspace(1.6, 2.3, 25) + np.random.normal(0, 0.15, 25), 1.1, 3.2)
patient3_df['Phosphate'] = np.clip(accelerated_linspace(3, 5.2, 25) + np.random.normal(0, 0.3, 25), 2, 6.5)
patient3_df['Potassium'] = np.clip(accelerated_linspace(3.7, 6.5, 25) + np.random.normal(0, 0.35, 25), 3, 7.5)
patient3_df['Bilirubin_total'] = np.clip(accelerated_linspace(0.55, 3.3, 25) + np.random.normal(0, 0.4, 25), 0.4, 5)
patient3_df['TroponinI'] = np.clip(accelerated_linspace(0, 0.5, 25) + np.random.normal(0, 0.05, 25), 0, 1.2)
patient3_df['Hct'] = np.clip(accelerated_linspace(43, 30, 25) + np.random.normal(0, 2, 25), 27, 48)
patient3_df['Hgb'] = np.clip(accelerated_linspace(14.5, 9.5, 25) + np.random.normal(0, 0.5, 25), 7.5, 16)
patient3_df['PTT'] = np.clip(accelerated_linspace(30, 60, 25) + np.random.normal(0, 4, 25), 18, 75)
patient3_df['WBC'] = np.clip(accelerated_linspace(6, 25, 25) + np.random.normal(0, 3, 25), 3, 30)
patient3_df['Fibrinogen'] = np.clip(accelerated_linspace(275, 540, 25) + np.random.normal(0, 30, 25), 180, 650)
patient3_df['Platelets'] = np.clip(accelerated_linspace(260, 125, 25) + np.random.normal(0, 15, 25), 90, 380)

patient3_df['Age'] = 62
patient3_df['Gender'] = 1
patient3_df['Unit1'] = 1
patient3_df['Unit2'] = 0
patient3_df['HospAdmTime'] = -85
patient3_df['ICULOS'] = patient3_df['Hour']


patient3_df['Patient_ID'] = 3


# Round values for realism
cols_to_round = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
                 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
                 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

for col in cols_to_round:
    patient3_df[col] = patient3_df[col].round(2)

print(patient3_df.head(15))
patient3_df.to_csv('patient3_sepsis_simulated.csv', index=False)