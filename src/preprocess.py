import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_and_normalize(df):
    print("🛠️ Memulai Preprocessing...")
    
    # 1. Handling Missing Values
    df = df.dropna()
    
    # 2. Encode Label (Mengubah nama serangan jadi angka)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    # 3. Fitur Scaling (Biar model SLGRAE-ENet lo gak pusing liat angka gede)
    # Kita pisahkan label dari fitur dulu
    X = df.drop(columns=['label'])
    y = df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Preprocessing Beres. {X_scaled.shape[1]} fitur siap tempur.")
    return X_scaled, y, le