records = []
for i in range(len(df)) :
    row, col = df.iloc[i,0].shape
    extent = row * col
    records.append(extent)
print(np.mean(records))