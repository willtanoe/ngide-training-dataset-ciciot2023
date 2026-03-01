import torch
import joblib
from train_enet import ENetClassifier
from sklearn.metrics import classification_report, accuracy_score

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data Laten & Model
    data = joblib.load('project_ids/data/processed/latent_features.pkl')
    X_test = torch.FloatTensor(data['X_test']).to(device)
    y_test = data['y_test']
    
    model = ENetClassifier(input_dim=16, num_classes=34).to(device)
    model.load_state_dict(torch.load('project_ids/models/enet_classifier.pth'))
    model.eval()

    # 2. Predict
    print("🧠 Evaluating model on test set...")
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()

    # 3. Hasil
    print(f"🎯 Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data['classes']))

if __name__ == "__main__":
    evaluate()