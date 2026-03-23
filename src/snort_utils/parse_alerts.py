import pandas as pd

alert_file = "alert_csv.txt"

with open(alert_file, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 4:
            break