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
out_dir1 = '/Users/chloe/Desktop/mean_rt.png' 
out_dir2 = '/Users/chloe/Desktop/median_rt.png'
out_dir3 = '/Users/chloe/Desktop/dist_rt.png'

data = [] # path data
all_succ1 = [] # sample data for all instances, success without restart
all_succ2 = [] # sample data for all instances, success with restart
# opt_len = [0]*len(all_instances) # optimal length for all instances

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
	rt = float(line['rt'])
	optlen = int(line['optimal_length'])
	ins_index = all_instances.index(instance)
	# opt_len[ins_index] = optlen
	if complete == 'True':
		if prev_sub == subject and prev_ins == instance: # success but restarted before
			all_succ2[ins_index].append(rt)
		else: # first trial success
			all_succ1[ins_index].append(rt)
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
ax[0].bar(np.arange(len(all_instances)), mean_succ1, alpha=0.5, color='green')
ax[0].set_ylim(top=190000)
ax[0].errorbar(np.arange(len(all_instances)), mean_succ1, \
	yerr=stderror_succ1, alpha=0.95, c='olive', fmt='none', capsize=1.5)
ax[0].yaxis.set_major_locator(MaxNLocator(21))
ax[0].grid(axis = 'y', alpha = 0.3)
ax[0].set_facecolor('0.98')
ax[0].set_ylabel('Response Time (ms)')
ax[0].set_title('Success Without Restart')
plt.sca(ax[0])
plt.xticks([0,18,36,53], ['7','11','14','16'])

ax[1].bar(np.arange(len(all_instances)), mean_succ2, alpha=0.7, color='orange')
# ax[1].set_ylim(top=90)
ax[1].errorbar(np.arange(len(all_instances)), mean_succ2, \
	yerr=stderror_succ2, alpha=0.4, c='red', fmt='none', capsize=1.5)
ax[1].yaxis.set_major_locator(MaxNLocator(21))
ax[1].grid(axis = 'y', alpha = 0.3)
ax[1].set_facecolor('0.98')
ax[1].set_ylabel('Response Time (ms)')
ax[1].set_title('Success With Restart')
plt.sca(ax[1])
plt.xticks([0,18,36,53], ['7','11','14','16'])

fig.text(0.5, 0.029, \
	'Puzzles (Ordered by Optimal Length)', \
	ha='center')
plt.suptitle('Mean Human Response Time')
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

fig.text(0.5, 0.029, \
	'Puzzles (Ordered by Optimal Length)', \
	ha='center')
plt.suptitle('Median Human Response Time')
plt.savefig(out_dir2)
plt.close()



