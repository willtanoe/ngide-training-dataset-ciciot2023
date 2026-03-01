import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import pandas as pd
import numpy as np

def balancing_data_smart(input_path):
    print("📂 Loading prepared features...")
    data = joblib.load(input_path)
    X, y = data['X'], data['y']
    
    # 1. Train-Test Split dulu biar data test murni
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Clear memory data awal
    del X, y 
    
    print("✂️ Downsampling kelas mayoritas biar RAM gak meledak...")
    # Kita limit tiap kelas maksimal 50,000 sampel (udah cukup banget buat deep learning)
    unique_classes = np.unique(y_train)
    X_sampled_list = []
    y_sampled_list = []
    
    for cls in unique_classes:
        X_cls = X_train[y_train == cls]
        y_cls = y_train[y_train == cls]
        
        if len(X_cls) > 50000:
            X_s, y_s = resample(X_cls, y_cls, replace=False, n_samples=50000, random_state=42)
        else:
            X_s, y_s = X_cls, y_cls
            
        X_sampled_list.append(X_s)
        y_sampled_list.append(y_s)
    
    X_train_s = np.vstack(X_sampled_list)
    y_train_s = np.concatenate(y_sampled_list)
    
    print(f"⚖️ Applying SMOTE pada data yang sudah di-downsample...")
    # Sekarang bebannya jauh lebih ringan buat RAM 16GB lo
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = sm.fit_resample(X_train_s, y_train_s)
    
    print(f"✅ Berhasil! Ukuran data balanced: {X_res.shape}")
    
    final_bundle = {
        'X_train': X_res, 'y_train': y_res,
        'X_test': X_test, 'y_test': y_test,
        'classes': data['classes']
    }
    
    joblib.dump(final_bundle, 'project_ids/data/processed/split_balanced_data.pkl')
    print("📦 Data siap tempur disimpan!")

if __name__ == "__main__":
    balancing_data_smart('project_ids/data/processed/final_training_data.pkl')