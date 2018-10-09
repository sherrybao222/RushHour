# mean human length vs optimal length
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

json_file = '/Users/chloe/Documents/RushHour/data/paths.json'
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/length.png'
data = []
instance_dict = {}
mean_dict = {}
optimal_dict = {}
# preprocess dict
for i in range(len(all_instances)):
	instance_dict[all_instances[i] + '_count'] = 0
	instance_dict[all_instances[i]+'_len'] = 0
	mean_dict[all_instances[i]] = 0
	optimal_dict[all_instances[i]] = 0
# load json data
with open(json_file) as f:
	for line in f:
		data.append(json.loads(line))
# iterate through line
for i in range(0, len(data)):
	line = data[i]
	instance = line['instance']
	complete = line['complete']
	human_len = int(line['human_length'])
	opt_len = int(line['optimal_length'])
	if complete == 'False':
		continue
	else:
		instance_dict[instance + '_count'] += 1
		instance_dict[instance + '_len'] = instance_dict[instance + '_len'] + human_len
		optimal_dict[instance] = opt_len
# calculate mean
for i in range(len(all_instances)):
	if instance_dict[all_instances[i] + '_count'] == 0:
		mean_dict[all_instances[i]] = 0
		continue
	else:
		mean_dict[all_instances[i]] = instance_dict[all_instances[i] + '_len'] / instance_dict[all_instances[i] + '_count']
# print out result
print(mean_dict)
print(len(mean_dict))
print('\n')
print(mean_dict)
#print(list(map(float,mean_dict.values())))
# generate figure
#pyplot.hist(list(map(float,mean_dict.values())), color='brown')
#pyplot.show()
fig = plt.figure()
ax = fig.add_subplot(111)
yvals1 = list(map(float,mean_dict.values())) # human data
yvals2 = list(map(float,optimal_dict.values())) # optimal
# # add seperation
# yvals1.insert(18,0)
# yvals1.insert(37,0)
# yvals1.insert(55,0)
# yvals2.insert(18,0)
# yvals2.insert(37,0)
# yvals2.insert(55,0)
rect = ax.bar(np.arange(len(mean_dict)), yvals1, alpha=0.9, color='orange', label='human')
rect = ax.bar(np.arange(len(optimal_dict)), yvals2, alpha=0.65, color='green', label='optimal')
ax.set_xticklabels([])
ax.yaxis.set_major_locator(MaxNLocator(20))
ax.grid(axis = 'y', alpha = 0.3)
ax.set_facecolor('0.98')
ax.set_xlabel('Puzzle')
ax.set_ylabel('#moves')
plt.title('Human Length V.S. Optimal Length')
# plt.hist(list(map(float,mean_dict.values())),bins=len(mean_dict), label='human')
# plt.hist(list(map(float,optimal_dict.values())),bins=len(optimal_dict), label='optimal')
plt.legend(loc='upper left')
plt.show()
#plt.savefig(out_dir)