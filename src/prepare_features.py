import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def prepare_data(input_path):
    print("Reading parquet file...")
    df = pd.read_parquet(input_path)
    
    # 1. Pisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['label'])
    y = df['label']
    
    # 2. Label Encoding (Nama Serangan -> Angka)
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Simpan encoder buat dipake pas testing nanti
    joblib.dump(le, 'project_ids/data/processed/label_encoder.pkl')
    
    # 3. Scaling (StandardScaler biar mean=0, std=1)
    print("Scaling features (this might take a while)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simpan scaler biar data testing dapet perlakuan yang sama
    joblib.dump(scaler, 'project_ids/data/processed/scaler.pkl')
    
    # 4. Simpan hasil akhir yang sudah SIAP TRAINING
    print("Saving processed arrays...")
    # Kita simpan pake format joblib atau numpy biar load-nya instan pas training
    processed_data = {
        'X': X_scaled,
        'y': y_encoded,
        'classes': le.classes_
    }
    joblib.dump(processed_data, 'project_ids/data/processed/final_training_data.pkl')
    
    print("✅ SEMUA BERES! Data siap masuk ke SLGRAE-ENet.")
    print(f"Total Classes: {len(le.classes_)}")

if __name__ == "__main__":
    IN = "project_ids/data/processed/train_cleaned.parquet"
    prepare_data(IN)