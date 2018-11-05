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
	opt_len = 0
	# ['first success', 'restart 1', 'restart 1+', 'surrender']
	# first time success, success after 1 restar, success after 1+ restart, surrender
	# load json data
	with open(ins_path_file) as f:
		prev_sub = ''
		restart_once_list = []
		n_success = 0
		n_restart_once = 0
		n_restart_more = 0
		n_surrender = 0
		for line in f:
			line = json.loads(line)
			opt_len = int(line['optimal_length'])
			subject = line['subject']
			complete = line['complete']
			skipped = line['skipped']
			if skipped == 'True':
				n_surrender += 1
			if subject == prev_sub:
				if complete == 'True':
					if subject not in restart_once_list:
						n_restart_once += 1
						restart_once_list.append(subject)
					else:
						n_restart_more += 1 
			elif complete == 'True': # first time success
				n_success += 1
			prev_sub = subject	
		ins_distrib[0] = n_success
		ins_distrib[1] = n_restart_once
		ins_distrib[2] = n_restart_more
		ins_distrib[3] = n_surrender

	# visualize distribution for current puzzle
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x_axis = ['first success', 'restart 1', 'restart 1+', 'surrender']
	ax.bar(x_axis, ins_distrib, alpha=0.7, color='blue')
	#ax.set_xlabel('')
	ax.set_ylabel('#subjects')
	plt.title('Success/restart/surrender count, puzzle ' + instance + ', opt_len=' + str(opt_len))
	fig_out_dir = out_dir + instance + '/' + instance + '_rss_count.png'
	ax.grid(axis = 'y', alpha = 0.3)
	plt.savefig(fig_out_dir)
	plt.close()
