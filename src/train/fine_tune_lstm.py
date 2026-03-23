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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Klasör Yolları ve Loglama
ROOT = Path(__file__).resolve().parents[2]
logs_dir = ROOT / "logs" / "cross_eval"
models_dir = ROOT / "models"
cic_raw_dir = ROOT / "data" / "raw" / "cicids2017"

logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "fine_tune_lstm.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def fine_tune_lstm_v2():
    logging.info("--- LSTM Gelişmiş İnce Ayar (v2) Süreci Başlıyor ---")

    # 2. Veri Hazırlama (Öncekiyle aynı)
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

    # 4. Train/Test Bölünmesi
    X_train_ft, X_test, y_train_ft, y_test = train_test_split(X, y, test_size=0.90, random_state=42, stratify=y)

    # 5. Scaler ve Reshape
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_train_ft = scaler.transform(X_train_ft)
    X_test = scaler.transform(X_test)

    X_train_ft = np.reshape(X_train_ft, (X_train_ft.shape[0], 1, X_train_ft.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # 6. Model Yükleme ve Callbacks Hazırlığı
    logging.info("Model yükleniyor ve callbacks hazırlanıyor...")
    model = load_model(models_dir / "best_lstm_model.h5")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Erken durdurma: 3 epoch boyunca val_loss iyileşmezse durdur
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # En iyi modeli kaydetme
    checkpoint = ModelCheckpoint(
        filepath=str(models_dir / "fine_tuned_lstm_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # 7. Uzatılmış Eğitim (20 Epoch)
    logging.info("LSTM Fine-tuning (20 Epoch) başlatılıyor...")
    model.fit(
        X_train_ft, y_train_ft,
        epochs=20,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # 8. Final Test
    logging.info("En iyi ağırlıklara sahip model ile test yapılıyor...")
    best_model = load_model(models_dir / "fine_tuned_lstm_model.h5")
    y_pred_prob = best_model.predict(X_test, batch_size=2048)
    y_pred = (y_pred_prob > 0.4).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Atak (1)'])

    logging.info(f"V2 Fine-Tuned LSTM Confusion Matrix:\n{cm}")
    logging.info(f"V2 Fine-Tuned LSTM Raporu:\n{cr}")
    logging.info("Süreç başarıyla tamamlandı.")

if __name__ == "__main__":
    fine_tune_lstm_v2()