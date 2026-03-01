import pandas as pd
import os

def clean_and_convert(input_path, output_path):
    print("🚀 Memulai proses cleaning data...")
    # Load data (Gunakan chunksize kalau RAM lo megap-megap, tapi 3060Ti harusnya aman)
    df = pd.read_csv(input_path)
    
    # 1. Hapus baris dengan nilai kosong (NaN) atau Infinity
    df.dropna(inplace=True)
    
    # 2. Drop kolom yang isinya konstan (nggak guna buat AI)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)
    
    # 3. Simpan ke Parquet (Format tempur riset S2)
    df.to_parquet(output_path, engine='pyarrow')
    print(f"✅ Selesai! Data bersih disimpan di: {output_path}")
    print(f"📊 Jumlah fitur tersisa: {len(df.columns)}")

if __name__ == "__main__":
    IN = "project_ids/data/raw/train.csv"
    OUT = "project_ids/data/processed/train_cleaned.parquet"
    os.makedirs("project_ids/data/processed", exist_ok=True)
    clean_and_convert(IN, OUT)