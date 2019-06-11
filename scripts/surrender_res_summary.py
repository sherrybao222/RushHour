# summary of portion of surrender and restart
# run with py27
import json, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import Counter
import scipy.stats as stats

json_file = '/Users/chloe/Documents/RushHour/exp_data/paths_filtered.json'
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Desktop/surr_rest_summary.png'
out_dir2 = '/Users/chloe/Desktop/surr_rest_summary2.png'

data = []
success_first = [0] * len(all_instances) # success in the first trial
success_more = [0] * len(all_instances) # success after one or more restart
surrender_first = [0] * len(all_instances) # surrender in the first trial
surrender_more = [0] * len(all_instances) # surrender after one or more restart

# load json data
with open(json_file) as f:
	for line in f:
		data.append(json.loads(line))

# iterate through line
prev_sub = ''
prev_instance = ''
for i in range(0, len(data)):
	line = data[i]
	subject = line['subject']
	instance = line['instance']
	complete = line['complete']
	skipped = line['skipped']
	if complete == 'False': # current trial failed
		if subject == prev_sub and instance == prev_instance: # has restarted before
			if skipped == 'True': # surrender
				surrender_more[all_instances.index(instance)] += 1
		else: # first trial
			if skipped == 'True': # surrender
				surrender_first[all_instances.index(instance)] += 1
	else: # current trial succeeded
		if subject == prev_sub and instance == prev_instance: # has restarted before
			success_more[all_instances.index(instance)] += 1
		else: # first trial
			success_first[all_instances.index(instance)] += 1
	prev_sub = subject	
	prev_instance = instance

# calculate proportion
count_total = [x+y+z+k for x,y,z,k in \
				zip(success_first, success_more, surrender_first, surrender_more)]
p_success_first = [float(x)/float(y) for x,y in zip(success_first, count_total)]
p_success_more = [float(x)/float(y) for x,y in zip(success_more, count_total)]
p_surrender_first = [float(x)/float(y) for x,y in zip(surrender_first, count_total)]
p_surrender_more = [float(x)/float(y) for x,y in zip(surrender_more, count_total)]

# split into 4 levels
yvals1 = p_success_first # success without restart
success11 = yvals1[:18]
success12 = yvals1[18:36]
success13 = yvals1[36:53]
success14 = yvals1[53:]
yvals2 = p_success_more # success with restart
success21 = yvals2[:18]
success22 = yvals2[18:36]
success23 = yvals2[36:53]
success24 = yvals2[53:]
yvals3 = p_surrender_first # surrender without restart
surrender11 = yvals3[:18]
surrender12 = yvals3[18:36]
surrender13 = yvals3[36:53]
surrender14 = yvals3[53:]
yvals4 = p_surrender_more # surrender with restart
surrender21 = yvals4[:18]
surrender22 = yvals4[18:36]
surrender23 = yvals4[36:53]
surrender24 = yvals4[53:]

# calculate summary of group
succ1_mean = np.array((np.mean(success11), np.mean(success12), np.mean(success13), np.mean(success14)))
succ2_mean = np.array((np.mean(success21), np.mean(success22), np.mean(success23), np.mean(success24)))
surr1_mean = np.array((np.mean(surrender11), np.mean(surrender12), np.mean(surrender13), np.mean(surrender14)))
surr2_mean = np.array((np.mean(surrender21), np.mean(surrender22), np.mean(surrender23), np.mean(surrender24)))

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
data = (succ1_mean, succ2_mean, surr1_mean, surr2_mean)
colors = ('green','orange','red','red')
groups = ('Success without Restart','Success with Restart','Surrender without Restart','Surrender with Restart')
counts = [1,2,3,4]
for data,color,group,count in zip(data,colors,groups,counts):
	x = np.arange(4)
	y = data
	if count == 1: # succ1
		ax.scatter(x, y, alpha=0.95,c=color,label=group)
		std = (np.std(success11),np.std(success12),np.std(success13),np.std(success14))
		ax.errorbar(x,y,yerr=std,alpha=0.95, c=color,capsize=4)
	if count == 2: # succ2
		ax.scatter(x, y, alpha=0.6,c=color,label=group)
		std = (np.std(success21),np.std(success22),np.std(success23),np.std(success24))
		ax.errorbar(x,y,yerr=std,alpha=0.6,c=color,capsize=4)
	if count == 3: # surr1
		ax.scatter(x, y, alpha=0.85,c=color,label=group)
		std = (np.std(surrender11),np.std(surrender12),np.std(surrender13),np.std(surrender14))
		ax.errorbar(x,y,yerr=std,alpha=0.85,c=color,capsize=4)
	if count == 4: # surr2
		ax.scatter(x, y, alpha=0.3,c=color,label=group)
		std = (np.std(surrender21),np.std(surrender22),np.std(surrender23),np.std(surrender24))
		ax.errorbar(x,y,yerr=std,alpha=0.3,c=color,capsize=4)

ax.yaxis.set_major_locator(MaxNLocator(12))
plt.xticks([0,1,2,3],['7','11','14','16'])
ax.set_xlabel('Puzzles (Ordered by Optimal Length)')
ax.set_ylabel('Number of Subjects Count')
plt.legend(loc='upper right', prop={'size': 7})
plt.title('Summary of Number of Success (First Trial), Restart and Surender')
#plt.show()
plt.savefig(out_dir)








# scatter plot 2
fig = plt.figure()
ax = fig.add_subplot(111)
surr_all = [int(x)+int(y) for x,y in zip(surrender_first, surrender_more)]
surr_all_mean = [float(x)+float(y) for x,y in zip(surr1_mean, surr2_mean)]
surrender_all1 = [float(x)+float(y) for x,y in zip(surrender11, surrender21)]
surrender_all2 = [float(x)+float(y) for x,y in zip(surrender12, surrender22)]
surrender_all3 = [float(x)+float(y) for x,y in zip(surrender13, surrender23)]
surrender_all4 = [float(x)+float(y) for x,y in zip(surrender14, surrender24)]
data = (succ1_mean, succ2_mean, surr_all_mean)
colors = ('green','orange','red')
groups = ('Success without Restart','Success with Restart','Surrender')
counts = [1,2,3]
for data,color,group,count in zip(data,colors,groups,counts):
	x = np.arange(4)
	y = data
	if count == 1: # succ1
		ax.scatter(x, y, alpha=0.95,c=color,label=group)
		std = (np.std(success11),np.std(success12),np.std(success13),np.std(success14))
		ax.errorbar(x,y,yerr=std,alpha=0.95, c=color,capsize=4)
	if count == 2: # succ2
		ax.scatter(x, y, alpha=0.6,c=color,label=group)
		std = (np.std(success21),np.std(success22),np.std(success23),np.std(success24))
		ax.errorbar(x,y,yerr=std,alpha=0.6,c=color,capsize=4)
	if count == 3: # surr
		ax.scatter(x, y, alpha=0.7,c=color,label=group)
		std = (np.std(surrender_all1),np.std(surrender_all2),np.std(surrender_all3),np.std(surrender_all4))
		ax.errorbar(x,y,yerr=std,alpha=0.7,c=color,capsize=4)

ax.yaxis.set_major_locator(MaxNLocator(11))
# plt.yticks([0,2,4,6,8], ['0.0','0.2','0.4','0.6','0.8'], fontsize=14)
ax.set_ylim(top=1.0)
plt.xticks([0,1,2,3],['7','11','14','16'], fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('')
ax.set_ylabel('')
plt.title('')
plt.savefig(out_dir2)


# chi squared test
freq1 = [np.sum(success_first[:18]), \
		np.sum(success_first[18:36]), \
		np.sum(success_first[36:53]), \
		np.sum(success_first[53:])]

freq2 = [np.sum(success_more[:18]), \
		np.sum(success_more[18:36]), \
		np.sum(success_more[36:53]), \
		np.sum(success_more[53:])]

freq3 = [np.sum(surr_all[:18]), \
		np.sum(surr_all[18:36]), \
		np.sum(surr_all[36:53]), \
		np.sum(surr_all[53:])]

print('Surrender: ', freq3)
all_freq = [freq1, freq2, freq3]
print('All freq: ', all_freq)
print(stats.chi2_contingency(all_freq))
