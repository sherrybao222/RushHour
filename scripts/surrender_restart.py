# number of surrender and restart for all puzzles
import json,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import Counter

json_file = '/Users/chloe/Documents/RushHour/exp_data/paths_filtered.json'
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Desktop/surr_rest.png'

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

# insert blank
p_success_first.insert(53,0)
p_success_first.insert(36,0)
p_success_first.insert(18,0)
p_success_more.insert(53,0)
p_success_more.insert(36,0)
p_success_more.insert(18,0)
p_surrender_first.insert(53,0)
p_surrender_first.insert(36,0)
p_surrender_first.insert(18,0)
p_surrender_more.insert(53,0)
p_surrender_more.insert(36,0)
p_surrender_more.insert(18,0)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
y12 = [x+y for x,y in zip(p_success_first, p_success_more)]
y123 = [x+y+z for x,y,z in zip(p_success_first, p_success_more, p_surrender_first)]
ax.bar(np.arange(len(p_success_first)), p_success_first, \
				alpha=0.95, color='green', label='Success without Restart')
ax.bar(np.arange(len(p_success_more)), p_success_more, \
				bottom=p_success_first,alpha=0.6, color='orange', label='Success after Restart')
ax.bar(np.arange(len(p_surrender_first)), p_surrender_first, \
				bottom=y12,alpha=0.8, color='red', label='Surrender without Restart')
ax.bar(np.arange(len(p_surrender_more)), p_surrender_more, \
				bottom=y123,alpha=0.4, color='red', label='Surrender after Restart')
ax.yaxis.set_major_locator(MaxNLocator(12))
# plt.xticks([0,18,36,53], ['7','11','14','16'])
plt.xticks([0,19,38,56], ['7','11','14','16'])
plt.xticks(ha='right')
ax.grid(axis = 'y', alpha = 0.3)
ax.set_facecolor('0.98')
ax.set_xlabel('Puzzles (Ordered by Optimal Length)')
ax.set_ylabel('Proportion of Subject')
plt.title('Proportion of Success and Surender, with and without Restart')
plt.legend(loc='lower right', prop={'size': 6})
# plt.show()
plt.savefig(out_dir)