import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.read_csv('data/train.csv')

print(df.columns)
print(df.iloc[:, 2:])
df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)

prof_rows = df[df['profanity'] == 1]
print(prof_rows)