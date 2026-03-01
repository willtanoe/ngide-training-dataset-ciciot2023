import torch
import joblib
import numpy as np
from model import SLGRAE

def transform_to_latent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using: {device}")

    # 1. Load Model & Data
    model = SLGRAE(input_dim=44).to(device)
    model.load_state_dict(torch.load('project_ids/models/slgrae_base.pth'))
    model.eval()

    print("📂 Loading balanced training data...")
    data = joblib.load('project_ids/data/processed/split_balanced_data.pkl')
    
    # 2. Proses Konversi (Lakukan per batch biar GPU gak meledak)
    def get_latent(X_data):
        X_tensor = torch.FloatTensor(X_data).to(device)
        with torch.no_grad():
            latent, _ = model(X_tensor)
        return latent.cpu().numpy()

    print("🧬 Transforming features to latent space...")
    X_train_latent = get_latent(data['X_train'])
    X_test_latent = get_latent(data['X_test'])

    # 3. Simpan dataset BARU yang sudah RINGKAS (16 Fitur)
    latent_bundle = {
        'X_train': X_train_latent,
        'y_train': data['y_train'],
        'X_test': X_test_latent,
        'y_test': data['y_test'],
        'classes': data['classes']
    }
    
    joblib.dump(latent_bundle, 'project_ids/data/processed/latent_features.pkl')
    print(f"✅ Selesai! Dataset laten (16 fitur) disimpan.")
    print(f"📊 Shape Baru: {X_train_latent.shape}")

if __name__ == "__main__":
    transform_to_latent()