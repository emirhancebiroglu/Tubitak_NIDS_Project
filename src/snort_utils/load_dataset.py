import pandas as pd

chunksize = 100000

chunks = pd.read_csv("merged.csv", chunksize=chunksize)

for chunk in chunks:
    print("Chunk shape:", chunk.shape)
    print("Label distribution in this chunk:\n", chunk[" Label"].value_counts())
    print(chunk.head())
    break  # sadece ilk chunk'u görmek için