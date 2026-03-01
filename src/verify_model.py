import torch
import joblib
import numpy as np
from model import SLGRAE

# 1. Load Model & Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SLGRAE(input_dim=44).to(device)
model.load_state_dict(torch.load('project_ids/models/slgrae_base.pth'))
model.eval()

data = joblib.load('project_ids/data/processed/split_balanced_data.pkl')
X_test = torch.FloatTensor(data['X_test'][:1000]).to(device) # Ambil 1000 sampel aja buat tes

# 2. Cek Hasil Rekonstruksi
with torch.no_grad():
    latent, reconstructed = model(X_test)
    
# Hitung Error (MSE) per sampel
mse = torch.mean((X_test - reconstructed)**2, dim=1).cpu().numpy()

print(f"📊 Average Reconstruction Error: {np.mean(mse):.6f}")
print(f"🧬 Latent Space Shape: {latent.shape}") # Harusnya (1000, 16)