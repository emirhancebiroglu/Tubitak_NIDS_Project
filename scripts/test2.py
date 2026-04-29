import xgboost as xgb
import pandas as pd
import numpy as np

X_test = np.load("../data/processed/X_test.npy")

model = xgb.Booster()
model.load_model("../models/fine_tuned_xgb_model.json")

# Modelin gördüğü feature names ve class encoding'i kontrol et
print("Feature names:", model.feature_names)
print("Num classes:", model.attr('num_class'))

# Test verinden birkaç bilinen ATTACK ve BENIGN örneği al
# X_attack = bilinen saldırı flowları
# X_benign = bilinen benign flowları

dmat_attack = xgb.DMatrix(X_attack[:10])
dmat_benign = xgb.DMatrix(X_benign[:10])

scores_attack = model.predict(dmat_attack)
scores_benign = model.predict(dmat_benign)

print(f"Bilinen ATTACK skorları: {scores_attack}")
print(f"Bilinen BENIGN skorları: {scores_benign}")