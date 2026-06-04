"""
automate_Ilyas-Lucky-Firmansyah.py
-----------------------------------
Script otomatisasi preprocessing dataset:
  Ultimate Memory Crisis Hardware Market Dynamics

Logika identik dengan notebook Eksperimen_Ilyas-Lucky-Firmansyah.ipynb.

Penggunaan:
    python automate_Ilyas-Lucky-Firmansyah.py \
        --input  ../memory_crisis_raw/datasets_raw.csv \
        --output memory_crisis_preprocessing
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing otomatis – Memory Crisis dataset"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path ke file CSV raw dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Direktori output untuk menyimpan hasil preprocessing",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_dir: str) -> None:

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"[1/9] Memuat dataset dari: {input_path}")
    df = pd.read_csv(input_path)
    print(f"      Shape awal: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Hapus baris DDR6 (Preview) — generasi belum stabil
    # ------------------------------------------------------------------
    print("[2/9] Menghapus baris DDR6 (Preview) ...")
    before = df.shape[0]
    df = df[df["generation"] != "DDR6 (Preview)"].copy()
    print(f"      {before - df.shape[0]} baris dihapus. Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 3. Drop kolom yang tidak informatif / data leakage
    # ------------------------------------------------------------------
    print("[3/9] Menghapus kolom ...")
    cols_to_drop = [
        "kit_id",               # ID unik, tidak informatif
        "model_name",           # terlalu spesifik / high-cardinality
        "timing_string",        # redundan (sudah ada cas_latency, speed_mts)
        "bandwidth_per_dollar", # leakage: dihitung dari price_usd
        "latency_value_index",  # leakage: korelasi 0.96 dengan target
        "speed_premium_ratio",  # leakage: korelasi 0.52 dengan target
        "price_per_gb",         # leakage: dihitung langsung dari price_usd
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"      Kolom tersisa: {list(df.columns)}")
    print(f"      Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 4. Tangani missing values & duplikat
    # ------------------------------------------------------------------
    print("[4/9] Memeriksa missing values & duplikat ...")
    missing = df.isnull().sum()
    if missing.any():
        print(f"      Missing values ditemukan:\n{missing[missing > 0]}")
        df.dropna(inplace=True)
        print(f"      Shape setelah drop NA: {df.shape}")
    else:
        print("      Tidak ada missing values.")

    n_dup = df.duplicated().sum()
    print(f"      Duplikat: {n_dup}")
    df.drop_duplicates(inplace=True)
    print(f"      Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 5. Capping outlier — persentil 1–99 untuk kolom numerik kontinu
    #    (bukan target, bukan hasil OHE)
    # ------------------------------------------------------------------
    print("[5/9] Capping outlier (1st–99th percentile) ...")
    numerical_cols_current = df.select_dtypes(include="number").columns.tolist()
    target = "price_usd"
    # Kita TIDAK cap target; is_ecc diabaikan karena biner
    cols_for_capping = [
        c for c in numerical_cols_current
        if c != target and c != "is_ecc"
    ]
    for col in cols_for_capping:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)
        print(f"      {col:35s}: [{lo:.4f}, {hi:.4f}]")
    print("      Capping selesai.")

    # ------------------------------------------------------------------
    # 6. Feature engineering
    # ------------------------------------------------------------------
    print("[6/9] Feature engineering ...")

    # is_ecc: bool → int
    df["is_ecc"] = df["is_ecc"].astype(int)

    # timestamp → year + month, lalu drop
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"]      = df["timestamp"].dt.year
    df["month"]     = df["timestamp"].dt.month
    df.drop(columns=["timestamp"], inplace=True)
    print(f"      Ditambahkan: year, month. Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 7. One-Hot Encoding kolom kategorikal
    # ------------------------------------------------------------------
    print("[7/9] One-Hot Encoding ...")
    categorical_remaining = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    print(f"      Kolom kategorikal: {categorical_remaining}")

    df = pd.get_dummies(df, columns=categorical_remaining, drop_first=False)
    # Pastikan dtype hasil OHE adalah int (bukan bool)
    ohe_cols = [
        c for c in df.columns
        if c not in numerical_cols_current + ["year", "month"]
        and c != target
    ]
    df[ohe_cols] = df[ohe_cols].astype(int)
    print(f"      Shape setelah encoding: {df.shape}")

    # ------------------------------------------------------------------
    # 8. Train–test split
    # ------------------------------------------------------------------
    print("[8/9] Train-test split (80/20, random_state=42) ...")
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"      y_train: {y_train.shape}, y_test: {y_test.shape}")

    # ------------------------------------------------------------------
    # 8b. Scaling kolom numerik kontinu (fit hanya pada train)
    # ------------------------------------------------------------------
    print("      Scaling numerik kontinu ...")
    num_cols_to_scale = [
        "capacity_gb",
        "module_count",
        "speed_mts",
        "cas_latency",
        "voltage",
        "true_latency_ns",
        "global_inventory_weeks",
        "fab_utilization_rate",
        "gpu_hbm_trend_gb",
        "year",
        "month",
    ]
    # Hanya scale kolom yang ada di X_train (antisipasi dataset berbeda)
    scale_cols = [c for c in num_cols_to_scale if c in X_train.columns]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
    print(f"      Kolom di-scale: {scale_cols}")

    # ------------------------------------------------------------------
    # 9. Simpan hasil
    # ------------------------------------------------------------------
    print(f"[9/9] Menyimpan hasil ke: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"),  index=False)

    print(f"      X_train.csv : {X_train.shape}")
    print(f"      X_test.csv  : {X_test.shape}")
    print(f"      y_train.csv : {y_train.shape}")
    print(f"      y_test.csv  : {y_test.shape}")
    print("Selesai.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output)
