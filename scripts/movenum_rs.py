'''
plot number of moves when subject restart/surrender
'''

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

df = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

restart7 = df[(df.restart==1) & (df.initial==7)]
restart11 = df[(df.restart==1) & (df.initial==11)]
restart14 = df[(df.restart==1) & (df.initial==14)]
restart16 = df[(df.restart==1) & (df.initial==16)]

fig, ax = plt.subplots(4,1, figsize=(10,7))

ax[0].hist(restart7.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[0].set_title('Level 7', fontsize=10)
ax[0].axvline(restart7.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[0].set_ylim([0, .7])
ax[1].hist(restart11.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[1].set_title('Level 11', fontsize=10)
ax[1].set_ylabel('Frequency', fontsize=14)
ax[1].axvline(restart11.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[1].set_ylim([0, .7])
ax[2].hist(restart14.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[2].set_title('Level 14', fontsize=10)
ax[2].axvline(restart14.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[2].set_ylim([0, .7])
ax[3].hist(restart16.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[3].set_title('Level 16', fontsize=10)
ax[3].set_xlabel('Number of Moves', fontsize=14)
ax[3].axvline(restart16.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[3].set_ylim([0, .7])

fig.tight_layout(pad=.8, rect=[0, 0.03, 1, 0.95])
fig.suptitle('Number of Moves When Restarted', fontsize=16)


plt.show()
plt.close()


surrender7 = df[(df.surrender==1) & (df.initial==7)]
surrender11 = df[(df.surrender==1) & (df.initial==11)]
surrender14 = df[(df.surrender==1) & (df.initial==14)]
surrender16 = df[(df.surrender==1) & (df.initial==16)]

fig, ax = plt.subplots(4,1, figsize=(10,7))

ax[0].hist(surrender7.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[0].set_title('Level 7', fontsize=10)
ax[0].axvline(surrender7.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[0].set_ylim([0, .7])
ax[1].hist(surrender11.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[1].set_title('Level 11', fontsize=10)
ax[1].set_ylabel('Frequency', fontsize=14)
ax[1].axvline(surrender11.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[1].set_ylim([0, .7])
ax[2].hist(surrender14.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[2].set_title('Level 14', fontsize=10)
ax[2].axvline(surrender14.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[2].set_ylim([0, .7])
ax[3].hist(surrender16.move_num, bins=80, range=(0, 80), density=True, align='left')
ax[3].set_title('Level 16', fontsize=10)
ax[3].set_xlabel('Number of Moves', fontsize=14)
ax[3].axvline(surrender16.move_num.median(), color='k', linestyle='dashed', linewidth=1)
# ax[3].set_ylim([0, .7])

fig.tight_layout(pad=.8, rect=[0, 0.03, 1, 0.95])
fig.suptitle('Number of Moves When Surrendered', fontsize=16)

plt.show()
plt.close()

