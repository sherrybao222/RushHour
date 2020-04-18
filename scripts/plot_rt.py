'''
plot response time for each subject by level
'''

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

df = pd.read_csv('/Users/chloe/Desktop/subjects/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9.csv')

level7 = df[(df.initial==16)]

all_rt = []
puzzle_rt = []
for index, row in df.iterrows():
    if row['event'] == 'start':
    	if puzzle_rt != []:
    		all_rt.append(puzzle_rt)
    	puzzle_rt = []
    	continue
    puzzle_rt.append(row['rt'])
all_rt.append(puzzle_rt)

all_rt = all_rt[1:]
print(all_rt)
minlen = float('inf')
for puzzle_rt in all_rt:
	if len(puzzle_rt) < minlen:
		minlen = len(puzzle_rt)
print(minlen)

minlen_rt = []
for cur_rt in all_rt:
	if len(cur_rt)>= minlen:
		minlen_rt.append(cur_rt[:minlen])
	else:
		minlen_rt.append(cur_rt)

print(minlen_rt)
minlen_rt = np.median(minlen_rt, axis=0)


plt.bar(x=range(minlen), height=minlen_rt)
plt.title('Median RT across Move Number, Level 7')
plt.xlabel('Move Number')
plt.ylabel('Median RT')
plt.show()