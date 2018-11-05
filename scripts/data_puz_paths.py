# arrange path data by puzzle
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats

pathfile = '/Users/chloe/Documents/RushHour/data/paths.json'
outfile = '/Users/chloe/Documents/RushHour/data/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_data = []
out_file = []

# initialize lists
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	out_file.append(outfile + instance + '_paths.json')
	out_data.append([])
# read entire path file
with open(pathfile) as f:
	for line in f:
		cur_line = json.loads(line)
		cur_ins = cur_line['instance']
		ins_index = all_instances.index(cur_ins)
		print(ins_index)
		# append current line to corresponding list index
		cur_data = out_data[ins_index]
		cur_data.append(cur_line)
		out_data[ins_index] =  cur_data
def sort_by_sub(d):
    '''a helper function for sorting'''
    return d['subject']
# write to file
for i in range(0, len(all_instances)):
    cur_data = out_data[i]
    cur_data = sorted(cur_data, key=sort_by_sub) # sort data by subject
    cur_file = out_file[i]
    cur_file = open(cur_file, 'w+')
    for j in range(0, len(cur_data)):
	    json.dump(cur_data[j], cur_file)
	    cur_file.write('\n')
	    