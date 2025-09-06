import pandas as pd
import numpy as np

# 24 hours range
hours = np.arange(0, 25)

# Initialize dataframe
sim_df = pd.DataFrame({'Hour': hours})

# Simulate different but plausible vitals & labs for sepsis progression
sim_df['HR'] = np.clip(np.sin(np.linspace(0, 3 * np.pi, 25)) * 15 + 90 + np.random.normal(0, 3, 25), 70, 140)            # Fluctuating heart rate
sim_df['O2Sat'] = np.clip(np.linspace(99, 88, 25) + np.random.normal(0, 0.8, 25), 80, 100)                               # O2 saturation dropping
sim_df['Temp'] = np.clip(np.linspace(36.8, 40, 25) + np.random.normal(0, 0.15, 25), 36, 41.5)                             # Fever rise
sim_df['SBP'] = np.clip(115 - np.sqrt(hours)*8 + np.random.normal(0, 3, 25), 70, 130)                                        # Nonlinear BP drop
sim_df['MAP'] = np.clip(85 - np.sqrt(hours)*7 + np.random.normal(0,2, 25), 55, 100)                                         # MAP dropping
sim_df['DBP'] = np.clip(75 - np.sqrt(hours)*5 + np.random.normal(0, 2, 25), 40, 90)                                         # DBP dropping
sim_df['Resp'] = np.clip(np.linspace(15, 35, 25) + np.random.normal(0, 1.2, 25), 12, 40)                                     # Respiratory rate rising
sim_df['EtCO2'] = np.clip(np.linspace(38, 18, 25) + np.random.normal(0, 2.5, 25), 10, 45)                                   # EtCO2 dropping
sim_df['BaseExcess'] = np.clip(-np.sqrt(hours)*3 + np.random.normal(0, 0.6, 25), -10, 3)                                      # Metabolic acidosis
sim_df['HCO3'] = np.clip(24 - np.sqrt(hours)*4 + np.random.normal(0, 0.5, 25), 10, 25)                                       # Bicarbonate dropping
sim_df['FiO2'] = np.clip(np.linspace(0.21, 0.65, 25), 0.21, 1)                                                               # Increasing oxygen support
sim_df['pH'] = np.clip(np.linspace(7.42, 7.15, 25) + np.random.normal(0, 0.025, 25), 7.0, 7.45)                             # Slight acidosis
sim_df['PaCO2'] = np.clip(np.linspace(38, 55, 25) + np.random.normal(0, 3, 25), 30, 70)                                      # CO2 rising
sim_df['SaO2'] = np.clip(np.linspace(99, 88, 25) + np.random.normal(0, 1, 25), 80, 100)                                      # O2 saturation dropping
sim_df['AST'] = np.clip(np.linspace(15, 120, 25) + np.random.normal(0, 12, 25), 10, 150)                                     # Liver enzyme rise
sim_df['BUN'] = np.clip(np.linspace(8, 55, 25) + np.random.normal(0, 4, 25), 5, 65)                                          # Kidney function decline
sim_df['Alkalinephos'] = np.clip(np.linspace(60, 180, 25) + np.random.normal(0, 15, 25), 40, 200)
sim_df['Calcium'] = np.clip(np.linspace(9.2, 7.0, 25) + np.random.normal(0, 0.25, 25), 6.5, 10)                             # Calcium drop
sim_df['Chloride'] = np.clip(np.linspace(98, 115, 25) + np.random.normal(0, 3, 25), 90, 120)
sim_df['Creatinine'] = np.clip(np.linspace(0.9, 3.2, 25) + np.random.normal(0, 0.3, 25), 0.7, 4)

sim_df['Bilirubin_direct'] = np.clip(np.linspace(0.05, 1.5, 25) + np.random.normal(0, 0.15, 25), 0.0, 3)
sim_df['Glucose'] = np.clip(np.linspace(85, 180, 25) + np.random.normal(0, 10, 25), 70, 220)
sim_df['Lactate'] = np.clip(np.linspace(0.9, 8, 25) + np.random.normal(0, 0.5, 25), 0.5, 12)
sim_df['Magnesium'] = np.clip(np.linspace(1.7, 2.2, 25) + np.random.normal(0, 0.12, 25), 1.2, 3)
sim_df['Phosphate'] = np.clip(np.linspace(3.2, 5.0, 25) + np.random.normal(0, 0.25, 25), 2.5, 6)
sim_df['Potassium'] = np.clip(np.linspace(3.9, 6.2, 25) + np.random.normal(0, 0.3, 25), 3.5, 7.2)
sim_df['Bilirubin_total'] = np.clip(np.linspace(0.6, 3.0, 25) + np.random.normal(0, 0.3, 25), 0.5, 4.5)
sim_df['TroponinI'] = np.clip(np.linspace(0, 0.45, 25) + np.random.normal(0, 0.03, 25), 0, 1.0)

sim_df['Hct'] = np.clip(np.linspace(42, 32, 25) + np.random.normal(0, 1.5, 25), 28, 45)
sim_df['Hgb'] = np.clip(np.linspace(14, 10, 25) + np.random.normal(0, 0.4, 25), 8, 15)
sim_df['PTT'] = np.clip(np.linspace(28, 56, 25) + np.random.normal(0, 3, 25), 20, 70)
sim_df['WBC'] = np.clip(np.linspace(7, 20, 25) + np.random.normal(0, 2, 25), 4, 25)
sim_df['Fibrinogen'] = np.clip(np.linspace(280, 520, 25) + np.random.normal(0, 25, 25), 200, 600)
sim_df['Platelets'] = np.clip(np.linspace(270, 140, 25) + np.random.normal(0, 12, 25), 100, 350)

sim_df['Age'] = 58
sim_df['Gender'] = 0
sim_df['Unit1'] = 0
sim_df['Unit2'] = 1
sim_df['HospAdmTime'] = -80
sim_df['ICULOS'] = sim_df['Hour']

sim_df['Patient_ID'] = 2

# Round for realism
cols_to_round = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
                 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

for col in cols_to_round:
    sim_df[col] = sim_df[col].round(2)

# To display first few rows
print(sim_df.head())
sim_df.to_csv('simulated_sepsis_data.csv', index=False)