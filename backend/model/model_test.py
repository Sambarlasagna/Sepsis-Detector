import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

# Old simpler GRU model with 1 layer, hidden_dim=64, no dropout or extra layers
class OldGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out).squeeze(1)
        return out

# New improved GRU model with 2 layers, hidden_dim=128, dropout, and fc layers
class NewGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out).squeeze(1)
        return out


SEQ_LENGTH = 6
FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_patient_dataset.csv')
OLD_MODEL_PATH = 'model_1_gru_time_to_sepsis.pth'  # Path to old model weights
NEW_MODEL_PATH = 'gru_time_to_sepsis.pth'          # Path to new model weights


def predict_compare_models(start_hour=SEQ_LENGTH - 1, end_hour=24):
    df = pd.read_csv(DATA_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    old_model = OldGRUModel(input_dim=len(FEATURES))
    old_model.load_state_dict(torch.load(OLD_MODEL_PATH, map_location=device))
    old_model.to(device)
    old_model.eval()

    new_model = NewGRUModel(input_dim=len(FEATURES))
    new_model.load_state_dict(torch.load(NEW_MODEL_PATH, map_location=device))
    new_model.to(device)
    new_model.eval()

    for hour in range(start_hour, end_hour + 1):
        patients_at_hour = df[df['Hour'] == hour]['Patient_ID'].unique()
        for patient_id in patients_at_hour:
            patient_data = df[df['Patient_ID'] == patient_id].sort_values('Hour').reset_index(drop=True)

            idx_list = patient_data[patient_data['Hour'] == hour].index.to_list()
            if not idx_list:
                continue
            end_idx = idx_list[0]
            start_idx = end_idx - SEQ_LENGTH + 1
            if start_idx < 0:
                continue

            window = patient_data.loc[start_idx:end_idx, FEATURES]
            if len(window) != SEQ_LENGTH:
                continue

            input_array = window.to_numpy(dtype=np.float32)
            input_tensor = torch.tensor(input_array).unsqueeze(0).to(device)

            with torch.no_grad():
                old_pred = old_model(input_tensor).item()
                new_pred = new_model(input_tensor).item()

            print(f"Patient {patient_id}, Hour {hour} - Old Model: {old_pred:.2f} hrs | New Model: {new_pred:.2f} hrs")


if __name__ == '__main__':
    predict_compare_models()

