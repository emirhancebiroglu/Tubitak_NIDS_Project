import pandas as pd
import numpy as np
import logging
import os
import pickle
from pathlib import Path

# Uyarıları gizleme
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# 1. Klasör Yolları
ROOT = Path(__file__).resolve().parents[2]
models_dir = ROOT / "models"
cic_raw_dir = ROOT / "data" / "raw" / "cicids2017"

logging.basicConfig(level=logging.INFO, format='%(message)s')

def find_optimum_threshold():
    logging.info("--- LSTM Optimum Eşik Analizi Başlıyor ---")

    # 2. Veri Yükleme ve Hazırlık (Fine-tune verisiyle aynı)
    csv_files = list(cic_raw_dir.glob("*.csv"))
    df_list = [pd.read_csv(f, low_memory=False, on_bad_lines='skip') for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()

    mapping = {
        'Flow Duration': 'dur', 'Total Fwd Packets': 'spkts', 'Total Backward Packets': 'dpkts',
        'Total Length of Fwd Packets': 'sbytes', 'Total Length of Bwd Packets': 'dbytes',
        'Fwd Packet Length Mean': 'smeansz', 'Bwd Packet Length Mean': 'dmeansz',
        'Init_Win_bytes_forward': 'swin', 'Init_Win_bytes_backward': 'dwin',
        'Fwd IAT Mean': 'sintpkt', 'Bwd IAT Mean': 'dintpkt'
    }
    df = df[list(mapping.keys()) + ['Label']].copy()
    df.rename(columns=mapping, inplace=True)
    df['dur'] /= 1e6
    df['sintpkt'] /= 1000.0
    df['dintpkt'] /= 1000.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    log_cols = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur', 'sintpkt', 'dintpkt']
    for col in log_cols:
        df[col] = np.log1p(df[col])

    feature_order = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 
                     'smeansz', 'dmeansz', 'swin', 'dwin', 'sintpkt', 'dintpkt']
    
    X = df[feature_order].values
    y = df['label'].values

    # Fine-tuning'de kullanılan aynı %90'lık test setini ayırıyoruz
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.90, random_state=42, stratify=y)

    # 3. Ölçeklendirme ve Model Yükleme
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = load_model(models_dir / "fine_tuned_lstm_model.h5")

    # 4. Olasılık Tahminleri
    logging.info("Olasılık tahminleri yapılıyor...")
    y_probs = model.predict(X_test_reshaped, batch_size=2048).flatten()

    # 5. Eşik Taraması (Threshold Sweep)
    thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    results = []

    logging.info(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    logging.info("-" * 55)

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        logging.info(f"{t:<12} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
        results.append((t, f1, recall, precision))

    # 6. En iyi eşiği seçme (F1-Skoru en yüksek olan)
    best_t, best_f1, best_rec, best_prec = max(results, key=lambda x: x[1])

    logging.info("-" * 55)
    logging.info(f"ÖNERİLEN OPTİMUM EŞİK: {best_t}")
    logging.info(f"Beklenen Recall: {best_rec:.4f}")
    logging.info(f"Beklenen Precision: {best_prec:.4f}")

    # Final Raporu Göster
    y_final_pred = (y_probs >= best_t).astype(int)
    logging.info("\nFinal Sınıflandırma Raporu (Optimum Eşik ile):")
    logging.info(classification_report(y_test, y_final_pred, target_names=['Normal (0)', 'Atak (1)']))

if __name__ == "__main__":
    find_optimum_threshold()