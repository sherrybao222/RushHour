# visualize number of success/restart/surrender for each puzzle (from puzzle path data)
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

datafile = '/Users/chloe/Documents/RushHour/data/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/'

# iterate through each puzzle
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	ins_path_file = datafile + instance + '_paths.json'
	ins_distrib = [0] * 4
	# ['first success', 'restart 1', 'restart 1+', 'surrender']
	# first time success, success after 1 restar, success after 1+ restart, surrender
	opt_len = 0
	sub_list = []
	prev_sub = ''
	restart_more_list = []
	n_success = 0
	n_restart_once = 0
	n_restart_more = 0
	n_surrender = 0
	# load json data
	with open(ins_path_file) as f:
		for line in f:
			line = json.loads(line)
			opt_len = int(line['optimal_length'])
			subject = line['subject']
			complete = line['complete']
			skipped = line['skipped']
			if complete == 'False':
				if skipped == 'True': # surrender
					n_surrender += 1
					sub_list.append(subject)
				elif subject == prev_sub and subject not in restart_more_list:
					restart_more_list.append(subject) # restart 2+
			else: # complete
				if subject != prev_sub: # first time success
					n_success += 1
					sub_list.append(subject)
				elif subject in restart_more_list: # success after 1 restart
					n_restart_more += 1
					sub_list.append(subject)
				else: # success after more restart
					n_restart_once += 1
					sub_list.append(subject)
			prev_sub = subject	
	ins_distrib[0] = n_success
	ins_distrib[1] = n_restart_once
	ins_distrib[2] = n_restart_more
	ins_distrib[3] = n_surrender
	# make portion
	ins_distrib = np.array(ins_distrib) / float(len(sub_list))
	print(sum(ins_distrib))
	# visualize distribution for current puzzle
	fig = plt.figure(figsize=(4.5,6))
	ax = fig.add_subplot(111)
	x_axis = ['1st success', 'restart 1', 'restart 2+', 'surrender']
	ax.bar(x_axis, ins_distrib, width=0.4, alpha=0.6, color='gray')
	plt.xticks(rotation=12, fontsize=14)
	# ax.set_ylabel('portion',fontsize=14)
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	plt.yticks(fontsize=14)
	plt.title('Success/restart/surrender distribution', fontsize=15)
	plt.suptitle(instance + ', opt_len=' + str(opt_len), fontsize=18)
	fig_out_dir = out_dir + instance + '/' + instance + '_rss_count.png'
	ax.grid(axis = 'y', alpha = 0.3)
	plt.savefig(fig_out_dir)
	plt.close()
