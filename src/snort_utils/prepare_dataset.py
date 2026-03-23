import pandas as pd

chunksize = 100000
df_iter = pd.read_csv("merged.csv", chunksize=chunksize)

for i, chunk in enumerate(df_iter):
    chunk.columns = chunk.columns.str.strip()
    
    columns_to_keep = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp",
        "Label"
    ]
    
    df_small_chunk = chunk[columns_to_keep].copy()
    
    # Dosyaya yaz
    mode = 'w' if i == 0 else 'a'
    header = True if i == 0 else False
    df_small_chunk.to_csv("merged_small.csv", mode=mode, index=False, header=header)
    
    print(f"Chunk {i+1} processed.", flush=True)

print("merged_small.csv has been created")