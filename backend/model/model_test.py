import os
import torch
import pandas as pd
import numpy as np

# GRU model definition (must match training)
import torch.nn as nn
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

# Settings
SEQ_LENGTH = 6
FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv')
MODEL_PATH = 'gru_time_to_sepsis.pth'  # Adjust if needed

def predict_for_valid_patient_at_hour(desired_hour):
    df = pd.read_csv(DATA_PATH)

    # Find all patients with data at desired_hour
    patients_with_hour = df[df['Hour'] == desired_hour]['Patient_ID'].unique()
    if len(patients_with_hour) == 0:
        raise ValueError(f"No patients found with data at hour {desired_hour}")

    for patient_id in patients_with_hour:
        patient_data = df[df['Patient_ID'] == patient_id].sort_values(by='Hour')
        target_idx_list = patient_data[patient_data['Hour'] == desired_hour].index.to_list()
        if not target_idx_list:
            continue
        target_idx = target_idx_list[0]

        # Check enough data for full 6-hour window
        window_start = target_idx - SEQ_LENGTH + 1
        if window_start < patient_data.index.min():
            continue  # Not enough data for this patient

        # Extract input window and true label
        window = df.loc[window_start:target_idx, FEATURES]
        true_time_to_sepsis = df.loc[target_idx, 'time_to_sepsis']

        input_array = window.to_numpy(dtype=np.float32)
        input_tensor = torch.tensor(input_array).unsqueeze(0)  # (1, seq_len, features)

        # Load model and predict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GRUModel(input_dim=len(FEATURES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_time_to_sepsis = output.item()

        print(f"Patient {patient_id}, Hour {desired_hour}")
        print(f"Predicted time to sepsis: {pred_time_to_sepsis:.2f} hours")
        print(f"True time to sepsis: {true_time_to_sepsis:.2f} hours")
        break  # Predict for first valid patient and exit

if __name__ == '__main__':
    TEST_HOURS = [247]  # hour for prediction
    for hour in TEST_HOURS:
        predict_for_valid_patient_at_hour(hour)
