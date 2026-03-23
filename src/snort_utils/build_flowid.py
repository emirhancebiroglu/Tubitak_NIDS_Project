import pandas as pd

alerts = pd.read_csv("alert_csv.txt", header=None)

# sadece IP ve TCP/UDP olanları al
alerts = alerts[alerts[1].notna()]
alerts = alerts[alerts[3].notna()]

src_ip = alerts[1]
src_port = alerts[2]
dst_ip = alerts[3]
dst_port = alerts[4]
protocol = alerts[5]

protocol_map = {
    "TCP": 6,
    "UDP": 17,
    "ICMP": 1
}

protocol_num = protocol.map(protocol_map)

flow_id = (
    src_ip.astype(str) + "-" +
    dst_ip.astype(str) + "-" +
    src_port.fillna(0).astype(int).astype(str) + "-" +
    dst_port.fillna(0).astype(int).astype(str) + "-" +
    protocol_num.fillna(0).astype(int).astype(str)
)

print("Alert flows:", len(flow_id))
print(flow_id.head(10))