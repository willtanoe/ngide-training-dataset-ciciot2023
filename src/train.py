import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from model import SLGRAE

def train_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on: {device}")

    # 1. Load Data Balanced yang baru lo buat
    print("📦 Loading balanced data...")
    data = joblib.load('project_ids/data/processed/split_balanced_data.pkl')
    
    # Ubah ke Tensor & Pindahkan ke GPU
    X_train = torch.FloatTensor(data['X_train']).to(device)
    
    # Autoencoder belajar dari data itu sendiri (self-supervised)
    dataset = TensorDataset(X_train, X_train)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    # 2. Inisialisasi Model & Optimizer
    model = SLGRAE(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    print("🚀 Starting Training...")
    for epoch in range(10): # Coba 10 epoch dulu
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            _, decoded = model(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(loader):.6f}")

    # 4. Simpan Model
    torch.save(model.state_dict(), 'project_ids/models/slgrae_base.pth')
    print("✅ Model saved!")

if __name__ == "__main__":
    train_engine()