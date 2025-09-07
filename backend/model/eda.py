import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------ Parameters ------------------------
SEQ_LENGTH = 6       # Sequence length for GRU input
MAX_TIME = 6         # Max time-to-sepsis label cap
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sepsis_dataset.csv')
MODEL_SAVE_PATH = 'gru_time_to_sepsis.pth'

# ------------------------ 1. Data Preprocessing ------------------------
def preprocess_sepsis_data(df):
    # Drop redundant columns if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Sort by Patient_ID and Hour to keep temporal order
    df = df.sort_values(['Patient_ID', 'Hour'])

    # Identify feature columns
    features = [col for col in df.columns if col not in ['SepsisLabel', 'Patient_ID', 'Hour']]

    # Fill missing data by forward/backward fill per patient, then median fill
    df[features] = df.groupby('Patient_ID')[features].transform(lambda g: g.ffill().bfill())
    df[features] = df[features].fillna(df[features].median())

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features, scaler

# ------------------------ 2. Create time_to_sepsis label ------------------------
def create_time_to_sepsis_label(df, patient_col='Patient_ID', hour_col='Hour', label_col='SepsisLabel', max_time=MAX_TIME):
    df = df.copy()
    df['time_to_sepsis'] = max_time  # initialize with max_time (censoring time)
    patients = df[patient_col].unique()

    for patient in patients:
        patient_data = df[df[patient_col] == patient]
        sepsis_hours = patient_data[patient_data[label_col] == 1][hour_col].values

        if len(sepsis_hours) == 0:
            continue

        sepsis_start = sepsis_hours[0]  # first hour SepsisLabel=1
        time_to_sepsis_vals = (sepsis_start - patient_data[hour_col]).clip(lower=0, upper=max_time).values
        df.loc[patient_data.index, 'time_to_sepsis'] = time_to_sepsis_vals

    return df

# ------------------------ 3. Create fixed-length sequences ------------------------
def create_sequences(df, patient_col='Patient_ID', hour_col='Hour', feature_cols=None, seq_length=SEQ_LENGTH):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [patient_col, hour_col, 'SepsisLabel', 'time_to_sepsis']]

    sequences = []
    labels = []

    patients = df[patient_col].unique()
    for patient in patients:
        patient_data = df[df[patient_col] == patient].sort_values(hour_col)
        features_array = patient_data[feature_cols].values
        time_to_sepsis_array = patient_data['time_to_sepsis'].values

        for start in range(len(patient_data) - seq_length + 1):
            end = start + seq_length
            seq_X = features_array[start:end]
            seq_y = time_to_sepsis_array[end - 1]  # label at sequence end
            sequences.append(seq_X)
            labels.append(seq_y)

    X = np.array(sequences)
    y = np.array(labels)
    return X, y

# ------------------------ 4. GRU Model Definition ------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last timestep output
        return self.fc(out).squeeze(1)  # output shape (batch,)

# ------------------------ 5. Load data and prepare training ------------------------
def main():
    print("Loading raw data...")
    df_raw = pd.read_csv(DATA_CSV_PATH)

    print("Preprocessing data...")
    df, features, scaler = preprocess_sepsis_data(df_raw)

    print("Creating time-to-sepsis labels...")
    df = create_time_to_sepsis_label(df)

    print("Generating sequences for modeling...")
    X, y = create_sequences(df, feature_cols=features, seq_length=SEQ_LENGTH)

    print(f"Total sequences: {len(X)}")

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                 torch.tensor(y_val, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    input_dim = len(features)
    model = GRUModel(input_dim, hidden_dim=64, num_layers=1, output_dim=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()  # MAE loss for regression

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training complete.")

if __name__ == '__main__':
    main()
