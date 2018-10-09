# number of surrender and restart
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import Counter

json_file = '/Users/chloe/Documents/RushHour/data/paths.json'
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/surr_rest.png'
data = []
success_dict = {}
surrender_dict = {}
restart_dict = {}
# preprocess dict
for i in range(len(all_instances)):
	success_dict[all_instances[i]] = 0
	surrender_dict[all_instances[i]] = 0
	restart_dict[all_instances[i]] = 0
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
	if subject == prev_sub and instance == prev_instance:
		if skipped == 'True':
			surrender_dict[instance] += 1
		elif complete == 'True':
			restart_dict[instance] += 1
	elif complete == 'True':
		success_dict[instance] += 1
	prev_sub = subject	
	prev_instance = instance
# print out result
print(success_dict)
print(len(success_dict))
print('\n')
print(restart_dict)
print(surrender_dict)
# plot
fig = plt.figure()
ax = fig.add_subplot(111)
yvals1 = list(map(int,success_dict.values())) # success
yvals1.insert(18,0)
yvals1.insert(37,0)
yvals1.insert(55,0)
yvals2 = list(map(int,restart_dict.values())) # restart
yvals2.insert(18,0)
yvals2.insert(37,0)
yvals2.insert(55,0)
yvals3 = list(map(int,surrender_dict.values())) # surrender
yvals3.insert(18,0)
yvals3.insert(37,0)
yvals3.insert(55,0)
y12 = {x: success_dict.get(x, 0) + restart_dict.get(x, 0) for x in success_dict.keys()}
y12 = list(map(int, y12.values()))
y12.insert(18,0)
y12.insert(37,0)
y12.insert(55,0)
print(yvals1)
print(yvals2)
print(y12)
rect = ax.bar(np.arange(len(success_dict) + 3), yvals1, alpha=0.7, color='green', label='success')
rect = ax.bar(np.arange(len(restart_dict) + 3), yvals2, bottom=yvals1,alpha=0.7, color='orange', label='restart')
rect = ax.bar(np.arange(len(surrender_dict) + 3), yvals3, bottom=y12,alpha=0.7, color='red', label='surrender')
ax.set_xticklabels([0, 7,11,14,16])
ax.yaxis.set_major_locator(MaxNLocator(20))
ax.xaxis.set_major_locator(MaxNLocator(5))
plt.xticks(ha='right')
ax.grid(axis = 'y', alpha = 0.3)
ax.set_facecolor('0.98')
ax.set_xlabel('optimal length')
ax.set_ylabel('#subjects')
plt.title('#success(first trial) #restart #surender')
# plt.hist(list(map(float,mean_dict.values())),bins=len(mean_dict), label='human')
# plt.hist(list(map(float,optimal_dict.values())),bins=len(optimal_dict), label='optimal')
plt.legend(loc='upper right')
#plt.show()
plt.savefig(out_dir)