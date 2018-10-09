# summary of portion of surrender and restart
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import Counter

json_file = '/Users/chloe/Documents/RushHour/data/paths.json'
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/surr_rest_summary.png'
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
fig = plt.figure()
ax = fig.add_subplot(111)

yvals1 = list(map(int,success_dict.values())) # success
success1 = yvals1[:18]
success2 = yvals1[18:36]
success3 = yvals1[36:53]
success4 = yvals1[53:]
yvals2 = list(map(int,restart_dict.values())) # restart
restart1 = yvals2[:18]
restart2 = yvals2[18:36]
restart3 = yvals2[36:53]
restart4 = yvals2[53:]
yvals3 = list(map(int,surrender_dict.values())) # surrender
surrender1 = yvals3[:18]
surrender2 = yvals3[18:36]
surrender3 = yvals3[36:53]
surrender4 = yvals3[53:]

# calculate summary of group
succ_mean = np.array((np.mean(success1), np.mean(success2), np.mean(success3), np.mean(success4)))
res_mean = np.array((np.mean(restart1), np.mean(restart2), np.mean(restart3), np.mean(restart4)))
surr_mean = np.array((np.mean(surrender1), np.mean(surrender2), np.mean(surrender3), np.mean(surrender4)))
# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
data = (succ_mean,res_mean,surr_mean)
colors = ('green','orange','red')
groups = ('success','restart','surrender')
counts = [1,2,3]
for data,color,group,count in zip(data,colors,groups,counts):
	x = np.arange(4)
	y = data
	print(y)
	ax.scatter(x, y, alpha=0.5,c=color,label=group)
	if count == 1:
		std = (np.std(success1),np.std(success2),np.std(success3),np.std(success4))
		ax.errorbar(x,y,yerr=std,alpha=0.7, c=color)
		print(succ_mean)
	if count == 2:
		std = (np.std(restart1),np.std(restart2),np.std(restart3),np.std(restart4))
		ax.errorbar(x,y,yerr=std,alpha=0.7,c=color)
		print(res_mean)
	if count == 3:
		std = (np.std(surrender1),np.std(surrender2),np.std(surrender3),np.std(surrender4))
		ax.errorbar(x,y,yerr=std,alpha=0.7,c=color)
		print(surr_mean)
ax.set_xticklabels([0,7,11,14,16])
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.set_xlabel('optimal length')
ax.set_ylabel('#subjects')
plt.legend(loc='upper right')
plt.title('summary of #success(first trial) #restart #surender')
#plt.show()
plt.savefig(out_dir)

# hitogram plot
# rect = ax.bar(np.arange(len(succ_mean)), succ_mean, alpha=0.7, color='green', label='success')
# rect = ax.bar(np.arange(len(res_mean)), res_mean, bottom=succ_mean,alpha=0.7, color='orange', label='restart')
# rect = ax.bar(np.arange(len(surr_mean)), surr_mean, bottom=(succ_mean + res_mean),alpha=0.7, color='red', label='surrender')
# ax.set_xticklabels([0, 7,11,14,16])
# #ax.yaxis.set_major_locator(MaxNLocator(20))
# ax.xaxis.set_major_locator(MaxNLocator(5))
# plt.xticks(ha='right')
# ax.grid(axis = 'y', alpha = 0.3)
# ax.set_facecolor('0.98')
# ax.set_xlabel('optimal length')
# ax.set_ylabel('#subjects')
# ax.errorbar(np.arange(len(succ_mean)), (succ_mean + res_mean + surr_mean), 0, [np.std(surrender1), np.std(surrender2), np.std(surrender3)])
# plt.title('#success(first trial) #restart #surender')
# plt.legend(loc='upper right')
# plt.show()
# #plt.savefig(out_dir)