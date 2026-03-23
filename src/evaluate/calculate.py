import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

with open("alert.txt") as f:
    alert_flows = set(line.strip() for line in f)

chunksize = 10000
cm_total = [[0, 0], [0, 0]]

for chunk in pd.read_csv("merged_small.csv", chunksize=chunksize):
    chunk['Predicted'] = chunk['Flow ID'].isin(alert_flows).astype(int)
    chunk['Label_binary'] = chunk['Label'].apply(lambda x: 0 if x == "BENIGN" else 1)

    cm = confusion_matrix(chunk['Label_binary'], chunk['Predicted'])

    cm_total[0][0] += cm[0][0]
    cm_total[0][1] += cm[0][1]
    cm_total[1][0] += cm[1][0]
    cm_total[1][1] += cm[1][1]

tn, fp, fn, tp = cm_total[0][0], cm_total[0][1], cm_total[1][0], cm_total[1][1]
print("Confusion Matrix:")
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")