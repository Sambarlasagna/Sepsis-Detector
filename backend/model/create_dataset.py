import numpy as np
import pandas as pd
import os

FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_patient_dataset_test.csv')

NUM_HOURS = 25  # hours 0-24
PATIENT_ID = 'synthetic_001'

def generate_realistic_synthetic_data(num_hours, num_features):
    t = np.arange(num_hours)
    data = np.random.normal(0, 0.3, (num_hours, num_features))  # baseline noise

    # Sepsis-relevant feature indices
    HR_idx = FEATURES.index('HR')
    MAP_idx = FEATURES.index('MAP')
    O2Sat_idx = FEATURES.index('O2Sat')
    WBC_idx = FEATURES.index('WBC')
    Lactate_idx = FEATURES.index('Lactate')
    pH_idx = FEATURES.index('pH')
    FiO2_idx = FEATURES.index('FiO2')

    sepsis_onset = 6  # obvious early onset

    for hour in t:
        if hour < sepsis_onset:
            data[hour, HR_idx] += 75
            data[hour, MAP_idx] += 90
            data[hour, O2Sat_idx] += 98
            data[hour, WBC_idx] += 7
            data[hour, Lactate_idx] += 1.2
            data[hour, pH_idx] += 7.4
            data[hour, FiO2_idx] += 21
        else:
            data[hour, HR_idx] += 75 + 6 * (hour - sepsis_onset)
            data[hour, MAP_idx] += 90 - 5 * (hour - sepsis_onset)
            data[hour, O2Sat_idx] += 98 - 2 * (hour - sepsis_onset)
            data[hour, WBC_idx] += 7 + 2 * (hour - sepsis_onset)
            data[hour, Lactate_idx] += 1.2 + 0.7 * (hour - sepsis_onset)
            data[hour, pH_idx] += 7.4 - 0.05 * (hour - sepsis_onset)
            data[hour, FiO2_idx] += 21 + 3 * (hour - sepsis_onset)

    return data

def main():
    raw_data = generate_realistic_synthetic_data(NUM_HOURS, len(FEATURES))

    # No scaling, keep raw values so the model sees spikes
    df = pd.DataFrame(raw_data, columns=FEATURES)
    df['Patient_ID'] = PATIENT_ID
    df['Hour'] = np.arange(NUM_HOURS)

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Synthetic test dataset saved to {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    main()
