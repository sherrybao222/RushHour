'''
plot number of moves when subject restart/surrender
'''

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

df = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
restart_rows = df.loc[df['restart'] == 1]
surrender_rows = df.loc[df['surrender'] == 1]

print(restart_rows[restart_rows.move_num==0].shape[0])
print(surrender_rows[surrender_rows.move_num<6].shape[0])
print(surrender_rows.move_num==0)

fig, ax = plt.subplots()

a_heights, a_bins = np.histogram(restart_rows['move_num'], bins=20)
b_heights, b_bins = np.histogram(surrender_rows['move_num'], bins=a_bins)

width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label='restart')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label='surrender')
plt.grid(alpha=0.5)
plt.legend()
plt.show()


restart_rows.hist(column='move_num', bins=20, edgecolor='black', linewidth=1)
plt.show()
surrender_rows.hist(column='move_num', bins=20, edgecolor='black', linewidth=1)
plt.show()
