import pandas as pd

alerts = pd.read_csv("alert_csv.txt", header=None, nrows=10)

print("Alert shape:")
print(alerts.shape)

print("\nFirst rows:")
print(alerts)