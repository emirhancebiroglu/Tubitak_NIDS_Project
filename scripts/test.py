import pandas as pd
import numpy as np
from pathlib import Path

csv_dir = Path("../data/raw/cicds-labels")
files = {
    'Wednesday': 'Wednesday-workingHours.pcap_ISCX.csv',
    'Friday_DDoS': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday_PortScan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday_Bot': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Tuesday': 'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Thursday': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
}

# Her max_packets değeri için kaç saldırı flow'u inference tetikler?
thresholds = [2, 4, 6, 8, 10, 15, 20]

print(f"{'Attack Type':<20} {'Total':>8}", end="")
for t in thresholds:
    print(f"  mp={t:>2}(%)", end="")
print()
print("-" * 90)

for name, fname in files.items():
    df = pd.read_csv(csv_dir / fname, low_memory=False, on_bad_lines='skip', encoding='latin-1')
    df.columns = df.columns.str.strip()
    attacks = df[df['Label'] != 'BENIGN'].copy()
    if len(attacks) == 0:
        continue
    attacks['total_pkts'] = attacks['Total Fwd Packets'] + attacks['Total Backward Packets']
    
    for label in attacks['Label'].unique():
        sub = attacks[attacks['Label'] == label]
        print(f"  {label:<20} {len(sub):>8}", end="")
        for t in thresholds:
            pct = (sub['total_pkts'] >= t).mean() * 100
            print(f"  {pct:>7.1f}%", end="")
        print()