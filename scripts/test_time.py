import pandas as pd

df = pd.read_csv('/Users/chloe/Desktop/timedict_new.csv')
print(df.columns)

for col in df.columns:
	df[col] = df[col]/df['MakeMove']

print(df.mean())

