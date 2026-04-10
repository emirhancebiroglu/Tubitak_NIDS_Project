#!/usr/bin/env python3
"""
export_scaler.py — RobustScaler parametrelerini C++ header formatına dönüştürür.

Kullanım:
    python export_scaler.py --scaler ../../models/scaler.pkl

Çıktı:
    1) Terminale C++ kopyala-yapıştır formatında median/iqr dizileri basar
    2) scaler_params.h dosyası oluşturur (opsiyonel, --output ile)

Not: Feature sırası prepare_lstmdataset.py ile aynı olmalı:
     dur, spkts, dpkts, sbytes, dbytes, smeansz, dmeansz, swin, dwin, sintpkt, dintpkt
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

# Eğitim sırasında kullanılan özellik sırası
FEATURE_ORDER = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'smeansz', 'dmeansz', 'swin', 'dwin', 'sintpkt', 'dintpkt'
]

def export_scaler(scaler_path: Path, output_path: Path = None):
    # 1. Scaler yükleme
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # RobustScaler'ın iç parametreleri:
    #   center_ = median değerleri (fit sırasında hesaplanır)
    #   scale_  = IQR değerleri (Q3 - Q1, fit sırasında hesaplanır)
    median_vals = scaler.center_
    iqr_vals = scaler.scale_

    assert len(median_vals) == len(FEATURE_ORDER), \
        f"Scaler {len(median_vals)} özellik bekliyor, ama {len(FEATURE_ORDER)} tanımlı!"

    # 2. Terminal çıktısı (doğrulama için)
    print("=" * 60)
    print("RobustScaler Parametreleri (Feature Sırası ile)")
    print("=" * 60)
    print(f"{'Feature':<12} {'Median':>14} {'IQR (scale)':>14}")
    print("-" * 42)
    for i, feat in enumerate(FEATURE_ORDER):
        print(f"{feat:<12} {median_vals[i]:>14.6f} {iqr_vals[i]:>14.6f}")

    # 3. C++ formatında çıktı
    def fmt_array(arr):
        """double dizisini C++ initializer list formatına çevir"""
        items = [f"{v:.10f}" for v in arr]
        # Her satıra 4 değer koy, okunabilirlik için
        lines = []
        for j in range(0, len(items), 4):
            chunk = ", ".join(items[j:j+4])
            lines.append("        " + chunk)
        return ",\n".join(lines)

    cpp_block = f"""
// ---------------------------------------------------------------
// RobustScaler parametreleri — export_scaler.py tarafından üretildi
// Kaynak: {scaler_path.name}
// Feature sırası: {', '.join(FEATURE_ORDER)}
// ---------------------------------------------------------------
static ScalerParams g_scaler_params = {{
    // median (center_)
    {{
{fmt_array(median_vals)}
    }},
    // iqr (scale_ = Q3 - Q1)
    {{
{fmt_array(iqr_vals)}
    }}
}};
"""

    print("\n" + "=" * 60)
    print("C++ KODU (ml_inspector.cc'deki placeholder'ın yerine yapıştırın)")
    print("=" * 60)
    print(cpp_block)

    # 4. Opsiyonel: header dosyası olarak kaydet
    if output_path:
        header_content = f"""// scaler_params.h — Otomatik üretildi (export_scaler.py)
// Manuel düzenlemeyin! Scaler değişirse scripti tekrar çalıştırın.
#ifndef ML_INSPECTOR_SCALER_PARAMS_H
#define ML_INSPECTOR_SCALER_PARAMS_H

#include "flow_tracker.h"
{cpp_block}
#endif // ML_INSPECTOR_SCALER_PARAMS_H
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header_content)
        print(f"Header dosyası kaydedildi: {output_path}")

    # 5. Doğrulama: Basit bir test vektörü ile Python vs C++ uyumu kontrol
    print("\n" + "=" * 60)
    print("DOĞRULAMA TESTİ")
    print("=" * 60)
    # Sıfır vektörü (tüm özellikler 0) ile test
    test_raw = np.zeros(len(FEATURE_ORDER))
    # log1p(0) = 0, yani log1p adımı sonucu değiştirmez
    # RobustScaler: (0 - median) / iqr
    test_scaled = (test_raw - median_vals) / iqr_vals
    print("Sıfır vektörü (log1p sonrası) için beklenen scaled değerler:")
    for i, feat in enumerate(FEATURE_ORDER):
        print(f"  {feat}: {test_scaled[i]:.10f}")
    print("\nBu değerleri C++ tarafında aynı girdi ile karşılaştırın.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RobustScaler → C++ export")
    parser.add_argument("--scaler", type=str, required=True,
                        help="scaler.pkl dosya yolu")
    parser.add_argument("--output", type=str, default=None,
                        help="Opsiyonel: scaler_params.h çıktı yolu")
    args = parser.parse_args()

    scaler_path = Path(args.scaler)
    if not scaler_path.exists():
        print(f"HATA: {scaler_path} bulunamadı!")
        exit(1)

    output_path = Path(args.output) if args.output else None
    export_scaler(scaler_path, output_path)