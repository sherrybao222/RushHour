# PLOTTING: characterize trialdata each move
# density histograms of mag size
# probability of error as a function of mag size
# py27 or python3
import sys, math
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as st
from matplotlib.ticker import MaxNLocator

all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_optlen = [7] * len(all_instances) # initialize puzzle difficulty level for each puzzle instance
ins_optlen[18:36]=[11]*len(ins_optlen[18:36])
ins_optlen[36:53]=[14]*len(ins_optlen[36:53])
ins_optlen[53:]=[16]*len(ins_optlen[53:])
optlen_consider = 7 # puzzle difficulty considered in this plot

moves_file = '/Users/yichen/Desktop/trialdata_valid_true_dist7_processed.csv' # input data file
out_file = '/Users/yichen/Desktop/mag_error_se.png' # output file


move_data = pd.read_csv(moves_file)

error_optlen = {} # count of error move for each optimal length
count_optlen = {} # count of any moves for each optimal length

hash_optlen = [] # array of optimal length with non-error move
hash_optlen_error = [] # array of optimal length with error move

max_optlen = ''


#################################### PROCESS DATA ###############################

for i in range(len(move_data)):
	row = move_data.loc[i, :]
	if i == 0: # first line, initialize data storage
		max_optlen = int(row['max_node'])
		subject = row['subject']
		error_optlen[subject]=[0]* (max_optlen + 1) # number of error moves at different optimal length
		count_optlen[subject]=[0] * (max_optlen + 1) # number of occurance for different optimal length
	cur_ins = row['instance'] # puzzle
	if ins_optlen[all_instances.index(cur_ins)] != optlen_consider: # only consider specific level
		continue 
	if count_optlen.get(row['subject'])==None: # new subject
		error_optlen[row['subject']]=[0]* (max_optlen + 1) # number of error moves at different optimal length
		count_optlen[row['subject']]=[0] * (max_optlen + 1) # number of occurance for different optimal length
	if row['event'] == 'start':
		optlen = row['node_human_static']
		continue
	error = int(row['error_made'])
	count_optlen[subject][optlen] += 1
	if error == 1:
		error_optlen[subject][optlen] += 1
		hash_optlen_error.append(int(optlen))
	else: # non error move
		hash_optlen.append(int(optlen))
	optlen = row['node_human_static'] # update optlen for next iteration
	subject = row['subject']

####################################### PLOTTING ##################################

# hitogram of error or nonerror moves given distance to goal (optimal length)
fig, ax = plt.subplots(2,1, figsize=(30, 40))
ax[0].hist(hash_optlen, bins=np.arange(max_optlen+1)-0.5,\
			density=True, align='mid', label='Non-Error', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].set_xlim(left=-1, right=14)
ax[0].axvline(np.median(hash_optlen), \
			color='gray', linestyle='dashed', linewidth=2.5)
# ax12 = ax[0].twinx()
# ax12.hist(hash_optlen_error, bins=np.arange(max_optlen+1)-0.5,\
# 			density=True, align='mid', label='Error', \
# 			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
# ax12.axvline(np.median(hash_optlen_error), \
# 		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].hist(hash_optlen_error, bins=np.arange(max_optlen+1)-0.5,\
			density=True, align='mid', label='Error', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0].axvline(np.median(hash_optlen_error), \
		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc='upper center', bbox_to_anchor=(0.75,0.9), prop={'size': 60})
# ax12.legend(loc='upper center', bbox_to_anchor=(0.75,0.81), prop={'size': 50})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=80)
# ax12.locator_params(nbins=5, axis='y')
# ax12.tick_params(axis='both', which='major', labelsize=60, colors='orangered')
ax[0].set_ylabel('Proportion of Positions', fontsize=80)
ax[0].set_xlabel('Distance to Goal', fontsize=80)
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_title('Histogram of Distance to Goal Given Error/Correct', \
				fontsize=30, weight='bold')
print('total optlen nonerror sample size: ', len(hash_optlen))
print('total optlen_error sample size: ', len(hash_optlen_error))



# probability of error given distance to goal
count_optlen_array = []
error_optlen_array = []
for key in count_optlen:
	count_optlen_array.append(count_optlen[key])
	error_optlen_array.append(error_optlen[key])
count_optlen = np.array(count_optlen_array, dtype=np.float32)
error_optlen = np.array(error_optlen_array, dtype=np.float32)
print('count_optlen.shape', count_optlen.shape)
print('error_optlen.shape', error_optlen.shape)
print('count_optlen', count_optlen)
print('error_optlen', error_optlen)


y_array = error_optlen/count_optlen # probability of error
print('y_array.shape', y_array.shape)
print('y_array', y_array)

y_array_filtered = [[] for i in range(max_optlen+1)]
for subject_idx in range(len(y_array)):
	subject_data = y_array[subject_idx]
	for optlen in range(len(subject_data)):
		if not np.isnan(subject_data[optlen]):
			y_array_filtered[optlen].append(subject_data[optlen])
# print('y_array_filtered[2:-23]', y_array_filtered[2:-23])
# y_array_filtered = y_array_filtered[2:-23]
# y_array_filtered = np.array(y_array_filtered[2:-23])
# print('y_array_filtered.shape', y_array_filtered.shape)
# print('y_array_filtered', y_array_filtered)

y_array_mean = np.array([np.mean(y) for y in y_array_filtered])
y_array_std = np.array([np.std(y) for y in y_array_filtered])
y_array_size = np.array([len(y) for y in y_array_filtered]) # sample size, number of subjects
print('y_array_mean', y_array_mean)
print('y_array_std', y_array_std)
print('y_array_size', y_array_size)

all_count_optlen = np.sum(count_optlen, axis=0) # collapse to calculate sample size for each optlen
cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=np.min(all_count_optlen), vmax=np.max(all_count_optlen))
colors = [cmap(normalize(value)) for value in all_count_optlen] # gradient the color by sample size

# z = st.norm.ppf(1-0.05/2) # error bar preparation (95%CI)
# CIup = (error_optlen+z**2/2.0)/(count_optlen+z**2) + (z/(count_optlen+z**2))*np.sqrt(error_optlen*(count_optlen-error_optlen)/count_optlen+z**2/4.0)
# CIlow = (error_optlen+z**2/2.0)/(count_optlen+z**2) - (z/(count_optlen+z**2))*np.sqrt(error_optlen*(count_optlen-error_optlen)/count_optlen+z**2/4.0)		

CIup = 1.96 * y_array_std/np.sqrt(y_array_size)
CIlow = 1.96 * y_array_std/np.sqrt(y_array_size)
SEup = y_array_std/np.sqrt(y_array_size)
SElow = y_array_std/np.sqrt(y_array_size)
print('np.array([CIlow, CIup])', np.array([CIlow, CIup]))

ax[1].bar(x=np.arange(max_optlen+1), \
		height=y_array_mean, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Error', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, color in zip(np.arange(max_optlen+1), y_array_mean, colors):
    ax[1].errorbar(pos, y, yerr=[SEup[pos]], capsize=14, color=color, linewidth=8)
ax[1].set_ylim(top=1)
ax[1].tick_params(axis='both', which='major', labelsize=80)
ax[1].set_ylabel('Probability of Error', fontsize=80)
ax[1].set_xlabel('Graph Size', fontsize=80)
ax[1].set_title('Probability of Error Given Graph Size', \
				fontsize=30, weight='bold')

# fig.text(0.5, 0.029, \
# 	'Graph Size', \
# 	ha='center', fontsize=40)
# plt.suptitle('Length-'+str(optlen_consider-1)+' Puzzle', \
# 	fontsize=40, weight='bold')
plt.savefig(out_file)
plt.close()

# hypothesis testing
print('\nerror: optlen')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_optlen, hash_optlen_error,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_optlen, hash_optlen_error,\
						equal_var=False, nan_policy='omit'))
print('Median Non-error: ', np.median(hash_optlen))
print('Median error: ', np.median(hash_optlen_error))
print('Mean Non-error: ', np.mean(hash_optlen))
print('Mean error: ', np.mean(hash_optlen_error))









sys.exit()