import pandas as pd

alerts = pd.read_csv("alert_csv.txt", header=None)

alerts = alerts[[1,2,3,4,5]]

alerts[2] = alerts[2].fillna(0).astype(float).astype(int)
alerts[4] = alerts[4].fillna(0).astype(float).astype(int)

# protocol map
proto_map = {
    "TCP":6,
    "UDP":17,
    "ICMP":1,
    "IP":0
}

def valid_ip(ip):
    if pd.isna(ip):
        return False
    if ip.startswith("224.") or ip == "255.255.255.255" or ":" in ip:
        return False
    return True

alerts[5] = alerts[5].map(proto_map).fillna(0).astype(int)

alerts = alerts[
    (alerts[1].notna()) & 
    (alerts[3].notna()) & 
    (alerts[2] != 0) & 
    (alerts[4] != 0) &
    alerts[1].apply(valid_ip) &
    alerts[3].apply(valid_ip)
].copy()

alerts["FlowID"] = (
    alerts[3].astype(str) + "-" +
    alerts[1].astype(str) + "-" +
    alerts[4].astype(str) + "-" +
    alerts[2].astype(str) + "-" +
    alerts[5].astype(str)
)

flowids = alerts["FlowID"]

print("Clean alert flows:", len(flowids))
print(flowids.head(10))

flowids.to_csv("alert_flowids_fixed_clean.txt", index=False, header=False)