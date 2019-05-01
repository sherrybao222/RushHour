# mean/median human length, optimal length, stderr
# mean/median human length distribution
# split by with or without restart
# run with py27
import json, math, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats
from compiler.ast import flatten

len_file = '/Users/chloe/Documents/RushHour/exp_data/paths_filtered.json'
ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir1 = '/Users/chloe/Desktop/mean_len.png' 
out_dir2 = '/Users/chloe/Desktop/median_len.png'
out_dir3 = '/Users/chloe/Desktop/dist_len.png'
out_dir4 = '/Users/chloe/Desktop/mean_len_stacked.png'

data = [] # path data
all_succ1 = [] # sample data for all instances, success without restart
all_succ2 = [] # sample data for all instances, success with restart
opt_len = [0]*len(all_instances) # optimal length for all instances

# preprocess list
for i in range(len(all_instances)):
	all_succ1.append([])
	all_succ2.append([])

# load path data
with open(len_file) as f:
	for line in f:
		data.append(json.loads(line))

# iterate through every subject trial
prev_sub = ''
prev_ins = ''
for i in range(0, len(data)): 
	line = data[i]
	subject = line['subject'] + ':' + line['assignment']
	instance = line['instance']
	complete = line['complete']
	human_len = int(line['human_length'])
	optlen = int(line['optimal_length'])
	ins_index = all_instances.index(instance)
	opt_len[ins_index] = optlen
	if complete == 'True':
		if prev_sub == subject and prev_ins == instance: # success but restarted before
			all_succ2[ins_index].append(human_len)
		else: # first trial success
			all_succ1[ins_index].append(human_len)
	prev_sub = subject
	prev_ins = instance




# calculate mean human len and std error
mean_succ1 = []
mean_succ2 = []
stderror_succ1 = []
stderror_succ2 = []
for i in range(0, len(all_instances)):
	mean_succ1.append(np.mean(all_succ1[i]))
	mean_succ2.append(np.mean(all_succ2[i]))
	stderror_succ1.append(stats.sem(all_succ1[i]))
	stderror_succ2.append(stats.sem(all_succ2[i]))

# plot mean
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(np.arange(len(all_instances)), mean_succ1, alpha=0.5, color='green', label='Human (Mean, SEM)')
ax[0].set_ylim(top=90)
ax[0].errorbar(np.arange(len(all_instances)), mean_succ1, \
	yerr=stderror_succ1, alpha=0.95, c='olive', fmt='none', capsize=1.5)
ax[0].bar(np.arange(len(all_instances)), opt_len, alpha=0.7, color='blue', label='Optimal')
ax[0].yaxis.set_major_locator(MaxNLocator(21))
ax[0].grid(axis = 'y', alpha = 0.3)
ax[0].set_facecolor('0.98')
ax[0].set_ylabel('Number of Moves')
ax[0].set_title('Success Without Restart')
ax[0].legend(loc='upper left')
plt.sca(ax[0])
plt.xticks([0,18,36,53], ['7','11','14','16'])

ax[1].bar(np.arange(len(all_instances)), mean_succ2, alpha=0.7, \
			color='orange', label='Human (Mean, SEM)')
ax[1].set_ylim(top=90)
ax[1].errorbar(np.arange(len(all_instances)), mean_succ2, \
	yerr=stderror_succ2, alpha=0.4, c='red', fmt='none', capsize=1.5)
ax[1].bar(np.arange(len(all_instances)), opt_len, alpha=0.7, color='blue', label='Optimal')
ax[1].yaxis.set_major_locator(MaxNLocator(21))
ax[1].grid(axis = 'y', alpha = 0.3)
ax[1].set_facecolor('0.98')
ax[1].set_ylabel('Number of Moves')
ax[1].set_title('Success With Restart')
ax[1].legend(loc='upper left')
plt.sca(ax[1])
plt.xticks([0,18,36,53], ['7','11','14','16'])

fig.text(0.5, 0.029, \
	'Puzzles (Ordered by Optimal Length)', \
	ha='center')
plt.suptitle('Mean Human Length and Optimal Length')
plt.savefig(out_dir1)
plt.close()



# calculate median human len and std error
median_succ1 = []
median_succ2 = []
stderror_succ1 = []
stderror_succ2 = []
for i in range(0, len(all_instances)):
	median_succ1.append(np.median(all_succ1[i]))
	median_succ2.append(np.median(all_succ2[i]))
	stderror_succ1.append(1.2533 * stats.sem(all_succ1[i]))
	stderror_succ2.append(1.2533 * stats.sem(all_succ2[i]))

# plot median
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(np.arange(len(all_instances)), median_succ1, alpha=0.5, color='green', label='Human (Median)')
ax[0].set_ylim(top=90)
ax[0].errorbar(np.arange(len(all_instances)), median_succ1, \
	yerr=stderror_succ1, alpha=0.95, c='olive', fmt='none', capsize=1.5)
ax[0].bar(np.arange(len(all_instances)), opt_len, alpha=0.7, color='blue', label='Optimal')
ax[0].yaxis.set_major_locator(MaxNLocator(21))
ax[0].grid(axis = 'y', alpha = 0.3)
ax[0].set_facecolor('0.98')
ax[0].set_ylabel('Number of Moves')
ax[0].set_title('Success Without Restart')
ax[0].legend(loc='upper left')
plt.sca(ax[0])
plt.xticks([0,18,36,53], ['7','11','14','16'])

ax[1].bar(np.arange(len(all_instances)), median_succ2, alpha=0.7, \
			color='orange', label='Human (Median)')
ax[1].set_ylim(top=90)
ax[1].errorbar(np.arange(len(all_instances)), median_succ2, \
	yerr=stderror_succ2, alpha=0.4, c='red', fmt='none', capsize=1.5)
ax[1].bar(np.arange(len(all_instances)), opt_len, alpha=0.7, color='blue', label='Optimal')
ax[1].yaxis.set_major_locator(MaxNLocator(21))
ax[1].grid(axis = 'y', alpha = 0.3)
ax[1].set_facecolor('0.98')
ax[1].set_ylabel('Number of Moves')
ax[1].set_title('Success With Restart')
ax[1].legend(loc='upper left')
plt.sca(ax[1])
plt.xticks([0,18,36,53], ['7','11','14','16'])

fig.text(0.5, 0.029, \
	'Puzzles (Ordered by Optimal Length)', \
	ha='center')
plt.suptitle('Median Human Length and Optimal Length')
plt.savefig(out_dir2)
plt.close()




# seperate human length data to 4 levels
level1_succ1 = flatten(all_succ1[:18]) # level of optlen 7
level2_succ1 = flatten(all_succ1[18:36]) # level of optlen 11
level3_succ1 = flatten(all_succ1[36:53]) # level of optlen 14
level4_succ1 = flatten(all_succ1[53:]) # level of optlen 16
level1_succ2 = flatten(all_succ2[:18]) # level of optlen 7
level2_succ2 = flatten(all_succ2[18:36]) # level of optlen 11
level3_succ2 = flatten(all_succ2[36:53]) # level of optlen 14
level4_succ2 = flatten(all_succ2[53:]) # level of optlen 16

# plot distribution of humanlen
fig, ax = plt.subplots(2, 4, figsize=(15, 6))
ax[0, 0].hist(level1_succ1, range=(7,30), bins=np.arange(start=7,stop=31,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 7', \
			color='green', edgecolor='green', alpha=0.5, width=1)
ax[0, 0].xaxis.set_ticks(np.arange(7, 30, 2))
ax[0, 0].axvline(np.mean(level1_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level1_succ1), 2)))
ax[0, 0].axvline(np.median(level1_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level1_succ1), 2)))
ax[0, 0].legend(loc='upper right', fontsize='small')

ax[0, 1].hist(level2_succ1, range=(11,50), bins=np.arange(start=11,stop=51,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 11', \
			color='green', edgecolor='green', alpha=0.5, width=1)
ax[0, 1].xaxis.set_ticks(np.arange(11, 50, 5))
ax[0, 1].axvline(np.mean(level2_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level2_succ1), 2)))
ax[0, 1].axvline(np.median(level2_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level2_succ1), 2)))
ax[0, 1].legend(loc='upper right', fontsize='small')

ax[0, 2].hist(level3_succ1, range=(14,85), bins=np.arange(start=14,stop=86,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 14', \
			color='green', edgecolor='green', alpha=0.5, width=1)
ax[0, 2].xaxis.set_ticks(np.arange(14, 85, 7))
ax[0, 2].axvline(np.mean(level3_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level3_succ1), 2)))
ax[0, 2].axvline(np.median(level3_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level3_succ1), 2)))
ax[0, 2].legend(loc='upper right', fontsize='small')

ax[0, 3].hist(level4_succ1, range=(16,120), bins=np.arange(start=16,stop=121,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 16', \
			color='green', edgecolor='green', alpha=0.5, width=1)
ax[0, 3].xaxis.set_ticks(np.arange(16, 120, 14))
ax[0, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0, 3].axvline(np.mean(level4_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level4_succ1), 2)))
ax[0, 3].axvline(np.median(level4_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level4_succ1), 2)))
ax[0, 3].legend(loc='upper right', fontsize='small')


ax[1, 0].hist(level1_succ2, range=(7,30), bins=np.arange(start=7,stop=31,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 7', \
			color='orange', edgecolor='orange', alpha=0.5, width=1)
ax[1, 0].xaxis.set_ticks(np.arange(7, 30, 2))
ax[1, 0].axvline(np.mean(level1_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level1_succ2), 2)))
ax[1, 0].axvline(np.median(level1_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level1_succ2), 2)))
ax[1, 0].legend(loc='upper right', fontsize='small')

ax[1, 1].hist(level2_succ2, range=(11,50), bins=np.arange(start=11,stop=51,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 11', \
			color='orange', edgecolor='orange', alpha=0.5, width=1)
ax[1, 1].xaxis.set_ticks(np.arange(11, 50, 5))
ax[1, 1].axvline(np.mean(level2_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level2_succ2), 2)))
ax[1, 1].axvline(np.median(level2_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level2_succ2), 2)))
ax[1, 1].legend(loc='upper right', fontsize='small')

ax[1, 2].hist(level3_succ2, range=(14,85), bins=np.arange(start=14,stop=86,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 14', \
			color='orange', edgecolor='orange', alpha=0.5, width=1)
ax[1, 2].xaxis.set_ticks(np.arange(14, 85, 7))
ax[1, 2].axvline(np.mean(level3_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level3_succ2), 2)))
ax[1, 2].axvline(np.median(level3_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level3_succ2), 2)))
ax[1, 2].legend(loc='upper right', fontsize='small')

ax[1, 3].hist(level4_succ2, range=(16,120), bins=np.arange(start=16,stop=121,step=1)-0.5, \
			density=False, align='mid', label='Level: Optlen 16', \
			color='orange', edgecolor='orange', alpha=0.5, width=1)
ax[1, 3].xaxis.set_ticks(np.arange(16, 120, 14))
ax[1, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[1, 3].axvline(np.mean(level4_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level4_succ2), 2)))
ax[1, 3].axvline(np.median(level4_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level4_succ2), 2)))
ax[1, 3].legend(loc='upper right', fontsize='small')

fig.text(0.51, 0.029, \
	'Human Length', \
	ha='center')
fig.text(0.5, 0.889, \
	'Success without Restart', \
	ha='center')
fig.text(0.5, 0.466, \
	'Success with Restart', \
	ha='center')
fig.text(0.09, 0.5, 'Count', \
	va='center', rotation='vertical')
plt.suptitle('Distribution of Human Length')
plt.savefig(out_dir3)
plt.close()




# plot stacked mean
g1 = np.random.normal(size=(len(mean_succ1[:18])))
g2 = np.random.normal(size=(len(mean_succ1[18:36])))
g3 = np.random.normal(size=(len(mean_succ1[36:53])))
g4 = np.random.normal(size=(len(mean_succ1[53:])))
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(1+g1, mean_succ1[:18], alpha=0.5, \
			color='green', label='Optimal Length 7')
ax.errorbar(1+g1, mean_succ1[:18], \
	yerr=stderror_succ1[:18], alpha=0.5, c='olive', fmt='none', capsize=3)

ax.scatter(10+g2, mean_succ1[18:36], alpha=0.5, \
			color='green', label='Optimal Length 11')
ax.errorbar(10+g2, mean_succ1[18:36], \
	yerr=stderror_succ1[18:36], alpha=0.5, c='olive', fmt='none', capsize=3)

ax.scatter(20+g3, mean_succ1[36:53], alpha=0.5, \
			color='green', label='Optimal Length 14')
ax.errorbar(20+g3, mean_succ1[36:53], \
	yerr=stderror_succ1[36:53], alpha=0.5, c='olive', fmt='none', capsize=3)

ax.scatter(30+g3, mean_succ1[53:], alpha=0.5, \
			color='green', label='Optimal Length 16')
ax.errorbar(30+g3, mean_succ1[53:], \
	yerr=stderror_succ1[53:], alpha=0.5, c='olive', fmt='none', capsize=3)
ax.axhline(y=7, linewidth=1.5, linestyle='--', color='olive', alpha=0.5)
ax.axhline(y=11, linewidth=1.5, linestyle='--', color='olive', alpha=0.5)
ax.axhline(y=14, linewidth=1.5, linestyle='--', color='olive', alpha=0.5)
ax.axhline(y=16, linewidth=1.5, linestyle='--', color='olive', alpha=0.5)
ax.yaxis.set_major_locator(MaxNLocator(5))

ax.grid(axis = 'y', alpha = 0.3)
ax.set_facecolor('0.98')
ax.set_ylabel('Mean Human Solution Length')
ax.set_title('Success With Restart')
plt.xticks([])
plt.suptitle('Mean Human Length and Optimal Length')
plt.savefig(out_dir4)
plt.close()