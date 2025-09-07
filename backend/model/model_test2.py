import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

# -------- GRU Model Definition --------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

# -------- Settings --------
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

def predict_for_patient_011093_hours():
    df = pd.read_csv(DATA_PATH)
    patient_id = '011093'
    patient_data = df[df['Patient_ID'] == patient_id].sort_values(by='Hour').reset_index(drop=True)
    model = GRUModel(input_dim=len(FEATURES))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    for last_hour in range(53, 60):  # Adjust as needed for your available data range
        window = patient_data[patient_data['Hour'].between(last_hour-SEQ_LENGTH+1, last_hour)].reset_index(drop=True)
        if len(window) != SEQ_LENGTH or \
           not (window['Hour'] == list(range(last_hour-SEQ_LENGTH+1, last_hour+1))).all():
            print(f"Insufficient data for prediction ending at hour {last_hour}")
            continue
        print('VALID DATA FOUND:', window['Hour'].tolist())
        input_array = window[FEATURES].to_numpy(dtype=np.float32)
        input_tensor = torch.tensor(input_array).unsqueeze(0).to(device)

        true_time_to_sepsis = window.iloc[-1]['time_to_sepsis'] if 'time_to_sepsis' in window.columns else max(0, 50 - last_hour)
        with torch.no_grad():
            pred_time_to_sepsis = model(input_tensor).item()
        print(f"\nPatient {patient_id}, Hour {last_hour}")
        print(f"Predicted time to sepsis: {pred_time_to_sepsis:.2f} hours")
        print(f"True time to sepsis: {true_time_to_sepsis:.2f} hours")


if __name__ == '__main__':
    predict_for_patient_011093_hours()
