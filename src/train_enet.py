import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

class ENetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ENetClassifier, self).__init__()
        # Arsitektur Linear yang efisien untuk Elastic Net
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training ENet on: {device}")

    # 1. Load Latent Features yang baru lo buat
    print("📦 Loading latent features...")
    data = joblib.load('project_ids/data/processed/latent_features.pkl')
    
    X_train = torch.FloatTensor(data['X_train']).to(device)
    y_train = torch.LongTensor(data['y_train']).to(device)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    # 2. Inisialisasi Model
    model = ENetClassifier(input_dim=16, num_classes=34).to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimizer Adam dengan L1 & L2 penalty (Elastic Net style)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5) 

    # 3. Training Loop
    print("🚀 Training Classifier...")
    for epoch in range(20):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/20], Loss: {total_loss/len(loader):.6f}")

    # 4. Simpan Model
    torch.save(model.state_dict(), 'project_ids/models/enet_classifier.pth')
    print("✅ ENet Classifier saved!")

if __name__ == "__main__":
    train_classifier()