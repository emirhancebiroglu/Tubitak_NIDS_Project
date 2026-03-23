import pandas as pd

df = pd.read_csv("merged_small.csv")
alerts = set(line.strip() for line in open("alert.txt"))

df_sample = df.head(20)

y_true = []
y_pred = []

for idx, row in df_sample.iterrows():
    label = 0 if row['Label'] == "BENIGN" else 1
    y_true.append(label)
    
    pred = 1 if row['Flow ID'] in alerts else 0
    y_pred.append(pred)

TP = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==1)
TN = sum(1 for t, p in zip(y_true, y_pred) if t==0 and p==0)
FP = sum(1 for t, p in zip(y_true, y_pred) if t==0 and p==1)
FN = sum(1 for t, p in zip(y_true, y_pred) if t==1 and p==0)

print("Mini-check Confusion Matrix (first 20 rows):")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

print("\nSample table for manual check:")
check_table = df_sample.copy()
check_table['Predicted'] = ['ATTACK' if p==1 else 'BENIGN' for p in y_pred]
print(check_table[['Flow ID','Label','Predicted']])