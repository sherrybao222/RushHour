# correlation between rt and human solution length
# split by with or without restart
# run with py27
import json, math, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats
from compiler.ast import flatten
from decimal import Decimal


len_file = '/Users/chloe/Documents/RushHour/exp_data/paths_filtered.json'
ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir1 = '/Users/chloe/Desktop/corr_rt_len.png' 

data = [] # path data
all_succ_rt = [] # rt sample data for all instances, all success records
all_succ1_rt = [] # rt sample data for all instances, success without restart
all_succ2_rt = [] # rt sample data for all instances, success with restart
all_succ_len = [] # human len sample data for all instances, all success records
all_succ1_len = [] # human len sample data for all instances, success without restart
all_succ2_len = [] # human len sample data for all instances, success with restart


# preprocess list
for i in range(len(all_instances)):
	all_succ_rt.append([])
	all_succ1_rt.append([])
	all_succ2_rt.append([])
	all_succ_len.append([])
	all_succ1_len.append([])
	all_succ2_len.append([])

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
	rt = float(line['rt'])
	humanlen = int(line['human_length'])
	ins_index = all_instances.index(instance)
	if complete == 'True':
		all_succ_len[ins_index].append(humanlen)
		all_succ_rt[ins_index].append(rt)
		if prev_sub == subject and prev_ins == instance: # success but restarted before
			all_succ2_rt[ins_index].append(rt)
			all_succ2_len[ins_index].append(humanlen)
		else: # first trial success
			all_succ1_rt[ins_index].append(rt)
			all_succ1_len[ins_index].append(humanlen)
	prev_sub = subject
	prev_ins = instance




# calculate mean rt and human len 
mean_succ_rt = []
mean_succ1_rt = []
mean_succ2_rt = []
mean_succ_len = []
mean_succ1_len = []
mean_succ2_len = []
for i in range(0, len(all_instances)):
	mean_succ_rt.append(np.mean(all_succ_rt[i]))
	mean_succ1_rt.append(np.mean(all_succ1_rt[i]))
	mean_succ2_rt.append(np.mean(all_succ2_rt[i]))
	mean_succ_len.append(np.mean(all_succ_len[i]))
	mean_succ1_len.append(np.mean(all_succ1_len[i]))
	mean_succ2_len.append(np.mean(all_succ2_len[i]))

# flatten data for all trials
flat_succ_rt = flatten(all_succ_rt)
flat_succ1_rt = flatten(all_succ1_rt)
flat_succ2_rt = flatten(all_succ2_rt)
flat_succ_len = flatten(all_succ_len)
flat_succ1_len = flatten(all_succ1_len)
flat_succ2_len = flatten(all_succ2_len)

# plot correlation
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].scatter(x=mean_succ_len, y=mean_succ_rt, alpha=0.5, color='blue')
ax[0, 0].yaxis.set_major_locator(MaxNLocator(5))
ax[0, 0].grid(axis = 'y', alpha = 0.3)
ax[0, 0].set_facecolor('0.98')
corr, p = stats.spearmanr(a=mean_succ_len,b=mean_succ_rt)
ax[0, 0].set_title('All Successful Instances\n'+ \
				'Spearman R='+ \
				str(round(corr, 2))+\
				', P=%.5e'%Decimal(p), fontsize=9)

ax[0, 1].scatter(x=mean_succ1_len, y=mean_succ1_rt, alpha=0.5, color='green')
ax[0, 1].yaxis.set_major_locator(MaxNLocator(5))
ax[0, 1].grid(axis = 'y', alpha = 0.3)
ax[0, 1].set_facecolor('0.98')
corr, p = stats.spearmanr(a=mean_succ1_len,b=mean_succ1_rt)
ax[0, 1].set_title('Successful Instances Without Restart\n'+ \
				'Spearman R='+ \
				str(round(corr, 2))+\
				', P=%.5e'%Decimal(p), fontsize=9)

ax[0, 2].scatter(x=mean_succ2_len, y=mean_succ2_rt, alpha=0.5, color='orange')
ax[0, 2].yaxis.set_major_locator(MaxNLocator(5))
ax[0, 2].grid(axis = 'y', alpha = 0.3)
ax[0, 2].set_facecolor('0.98')
corr, p = stats.spearmanr(a=mean_succ2_len,b=mean_succ2_rt)
ax[0, 2].set_title('All Successful Instances With Restart\n'+ \
				'Spearman R='+ \
				str(round(corr, 2))+\
				', P=%.5e'%Decimal(p), fontsize=9)

ax[1, 0].scatter(x=flat_succ_len, y=flat_succ_rt, alpha=0.5, color='blue')
ax[1, 0].yaxis.set_major_locator(MaxNLocator(5))
ax[1, 0].grid(axis = 'y', alpha = 0.3)
ax[1, 0].set_facecolor('0.98')
corr, p = stats.spearmanr(a=flat_succ_len,b=flat_succ_rt)
ax[1, 0].set_title('All Successful Trials\n'+ \
				'Spearman R='+ \
				str(round(corr, 2))+\
				', P=%.5e'%Decimal(p), fontsize=9)

ax[1, 1].scatter(x=flat_succ1_len, y=flat_succ1_rt, alpha=0.5, color='green')
ax[1, 1].yaxis.set_major_locator(MaxNLocator(5))
ax[1, 1].grid(axis = 'y', alpha = 0.3)
ax[1, 1].set_facecolor('0.98')
corr, p = stats.spearmanr(a=flat_succ1_len,b=flat_succ1_rt)
ax[1, 1].set_title('Successful Trials Without Restart\n'+ \
				'Spearman R='+ \
				str(round(corr, 2))+\
				', P=%.5e'%Decimal(p), fontsize=9)

ax[1, 2].scatter(x=flat_succ2_len, y=flat_succ2_rt, alpha=0.5, color='orange')
ax[1, 2].yaxis.set_major_locator(MaxNLocator(5))
ax[1, 2].grid(axis = 'y', alpha = 0.3)
ax[1, 2].set_facecolor('0.98')
corr, p = stats.spearmanr(a=flat_succ2_len,b=flat_succ2_rt)
ax[1, 2].set_title('Successful Trials With Restart\n'+ \
				'Spearman R='+ \
				str(round(corr, 2))+\
				', P=%.5e'%Decimal(p), fontsize=9)

fig.text(0.51, 0.029, \
	'Human Length', \
	ha='center')
fig.text(0.05, 0.5, 'Human Response Time (ms)', \
	va='center', rotation='vertical')
plt.suptitle('Correlation of Human Response Time and Human Length')
plt.savefig(out_dir1)
plt.close()


sys.exit()

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
ax[0].bar(np.arange(len(all_instances)), median_succ1, alpha=0.5, color='green')
ax[0].set_ylim(top=170000, bottom=0)
ax[0].errorbar(np.arange(len(all_instances)), median_succ1, \
	yerr=stderror_succ1, alpha=0.95, c='olive', fmt='none', capsize=1.5)
ax[0].yaxis.set_major_locator(MaxNLocator(21))
ax[0].grid(axis = 'y', alpha = 0.3)
ax[0].set_facecolor('0.98')
ax[0].set_ylabel('Response Time (ms)')
ax[0].set_title('Success Without Restart')
plt.sca(ax[0])
plt.xticks([0,18,36,53], ['7','11','14','16'])

ax[1].bar(np.arange(len(all_instances)), median_succ2, alpha=0.7, color='orange')
ax[1].set_ylim(top=170000, bottom=0)
ax[1].errorbar(np.arange(len(all_instances)), median_succ2, \
	yerr=stderror_succ2, alpha=0.4, c='red', fmt='none', capsize=1.5)
ax[1].yaxis.set_major_locator(MaxNLocator(21))
ax[1].grid(axis = 'y', alpha = 0.3)
ax[1].set_facecolor('0.98')
ax[1].set_ylabel('Response Time (ms)')
ax[1].set_title('Success With Restart')
plt.sca(ax[1])
plt.xticks([0,18,36,53], ['7','11','14','16'])

fig.text(0.51, 0.029, \
	'Human Response Time (ms)', \
	ha='center')
fig.text(0.5, 0.889, \
	'Success without Restart', \
	ha='center')
fig.text(0.5, 0.466, \
	'Success with Restart', \
	ha='center')
fig.text(0.09, 0.5, 'Count', \
	va='center', rotation='vertical')
plt.suptitle('Correlation of Human Response Time and Human Length')
plt.savefig(out_dir1)
plt.close()





# seperate rt data to 4 levels
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
ax[0, 0].hist(level1_succ1, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 7', \
			color='green', edgecolor='green', alpha=0.5)
ax[0, 0].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[0, 0].axvline(np.mean(level1_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level1_succ1), 2)))
ax[0, 0].axvline(np.median(level1_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level1_succ1), 2)))
ax[0, 0].legend(loc='upper right', fontsize='small')

ax[0, 1].hist(level2_succ1, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 11', \
			color='green', edgecolor='green', alpha=0.5)
ax[0, 1].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[0, 1].axvline(np.mean(level2_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level2_succ1), 2)))
ax[0, 1].axvline(np.median(level2_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level2_succ1), 2)))
ax[0, 1].legend(loc='upper right', fontsize='small')

ax[0, 2].hist(level3_succ1, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 14', \
			color='green', edgecolor='green', alpha=0.5)
ax[0, 2].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[0, 2].axvline(np.mean(level3_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level3_succ1), 2)))
ax[0, 2].axvline(np.median(level3_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level3_succ1), 2)))
ax[0, 2].legend(loc='upper right', fontsize='small')

ax[0, 3].hist(level4_succ1, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 16', \
			color='green', edgecolor='green', alpha=0.5)
ax[0, 3].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[0, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0, 3].axvline(np.mean(level4_succ1), color='olive', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level4_succ1), 2)))
ax[0, 3].axvline(np.median(level4_succ1), color='olive', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level4_succ1), 2)))
ax[0, 3].legend(loc='upper right', fontsize='small')


ax[1, 0].hist(level1_succ2, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 7', \
			color='orange', edgecolor='orange', alpha=0.5)
ax[1, 0].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[1, 0].axvline(np.mean(level1_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level1_succ2), 2)))
ax[1, 0].axvline(np.median(level1_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level1_succ2), 2)))
ax[1, 0].legend(loc='upper right', fontsize='small')

ax[1, 1].hist(level2_succ2, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 11', \
			color='orange', edgecolor='orange', alpha=0.5)
ax[1, 1].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[1, 1].axvline(np.mean(level2_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level2_succ2), 2)))
ax[1, 1].axvline(np.median(level2_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level2_succ2), 2)))
ax[1, 1].legend(loc='upper right', fontsize='small')

ax[1, 2].hist(level3_succ2, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 14', \
			color='orange', edgecolor='orange', alpha=0.5)
ax[1, 2].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[1, 2].axvline(np.mean(level3_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level3_succ2), 2)))
ax[1, 2].axvline(np.median(level3_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level3_succ2), 2)))
ax[1, 2].legend(loc='upper right', fontsize='small')

ax[1, 3].hist(level4_succ2, range=(0, 170000), bins=22, \
			density=False, align='left', label='Level: Optlen 16', \
			color='orange', edgecolor='orange', alpha=0.5)
ax[1, 3].xaxis.set_ticks(np.arange(0, 170000, 40000))
ax[1, 3].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[1, 3].axvline(np.mean(level4_succ2), color='orangered', \
				alpha=0.9, linestyle='-', linewidth=1,\
				label='Mean: '+str(round(np.mean(level4_succ2), 2)))
ax[1, 3].axvline(np.median(level4_succ2), color='orangered', \
				alpha=0.9, linestyle=':', linewidth=1, \
				label='Median: '+str(round(np.median(level4_succ2), 2)))
ax[1, 3].legend(loc='upper right', fontsize='small')

fig.text(0.51, 0.029, \
	'Human Response Time (ms)', \
	ha='center')
fig.text(0.5, 0.889, \
	'Success without Restart', \
	ha='center')
fig.text(0.5, 0.466, \
	'Success with Restart', \
	ha='center')
fig.text(0.09, 0.5, 'Count', \
	va='center', rotation='vertical')
plt.suptitle('Distribution of Human Response Time')
plt.savefig(out_dir3)
plt.close()