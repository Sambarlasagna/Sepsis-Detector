import os
import torch
import pandas as pd
import numpy as np
import time

# 1. Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LENGTH = 12  # <-- Set your actual model sequence length!

start_time = time.time()

FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_sepsis_dataset.csv')
MODEL_PATH = 'gru_time_to_sepsis.pth'  # already in current directory

# 2. Load Data
df = pd.read_csv(DATA_PATH)

# 3. Find the first window that ends with SepsisLabel == 1
idx = df[df['SepsisLabel'] == 1].index[0]+6
window_start = idx - SEQ_LENGTH + 1
window_end = idx + 1

if window_start < 0:
    raise ValueError("Not enough data for full sequence window. Try a higher-index sepsis case.")

window = df.iloc[window_start:window_end]
sample_input = window[FEATURES].to_numpy()  # shape: (seq_length, num_features)

# 4. (Optional) If you normalized features during training, apply same normalization here!
# For example:
# sample_input = (sample_input - feature_means) / feature_stds

# 5. Convert to tensor and batch
input_tensor = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, SEQ_LENGTH, n_features)

# 6. Load your GRU model architecture and weights
# ------ Edit below class/args to match your training code! ------
class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Use output from last timestep
        return self.fc(out)
# ---------------------------------------------------------------

INPUT_DIM = len(FEATURES)
HIDDEN_DIM = 64         # <-- set to your model's trained value!
NUM_LAYERS = 1          # <-- set to your model's trained value!
OUTPUT_DIM = 1

model = GRUModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 7. Predict
with torch.no_grad():
    pred_hours = model(input_tensor)
    pred_hours = pred_hours.cpu().numpy().flatten()[0]

print(f"Time taken: {time.time() - start_time:.2f} seconds")
print(f"Predicted time to sepsis: {pred_hours:.2f} hours")
print(f"Actual SepsisLabel at sequence end: {window.iloc[-1]['SepsisLabel']}")
