# save new path data by filtering out invalid subjects
# invalid subject: 1. surrender within 7 moves for every trial; 2. incomplete at the end
# save only successful trials
import json, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats

csvfile = '/Users/chloe/Documents/RushHour/exp_data/trialdata.csv'
movefile = '/Users/chloe/Documents/RushHour/exp_data/moves.json' # unfiltered file
pathfile = '/Users/chloe/Documents/RushHour/exp_data/paths.json' # unfiltered file
outfile = '/Users/chloe/Documents/RushHour/exp_data/moves_filtered.json'
outfile2 = '/Users/chloe/Documents/RushHour/exp_data/paths_filtered.json'
humanfile = '/Users/chloe/Documents/RushHour/state_model/in_data_trials/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
# valid includes bonus and postquestionare
valid_subjects = np.load('/Users/chloe/Documents/RushHour/exp_data/valid_sub.npy')
# valid_subjects2 = []
# for i in range(0, len(valid_subjects)):
# 	if valid_subjects[i] not in valid_subjects2:
# 		valid_subjects2.append(valid_subjects[i])
# valid_subjects = valid_subjects2
# print('valid subjects len', len(valid_subjects))
# np.save('/Users/chloe/Documents/RushHour/exp_data/valid_sub.npy', valid_subjects)


def filter_move():
	# read in move data
	old_move_data = []
	with open(movefile) as f: 
		for line in f:
			old_move_data.append(json.loads(line))
	# process move data
	new_data = []
	for i in range(0, len(old_move_data)): 
		line = old_move_data[i]
		subject = line['subject']
		if subject in valid_subjects:
			new_data.append(line)
	# save new move data
	result_sub = [] # check if the resulting number of subjects
	with open(outfile, 'w+') as f:
		for i in range(0, len(new_data)): # each line
			json.dump(new_data[i], f)
			if new_data[i]['subject'] not in result_sub:
				result_sub.append(new_data[i]['subject'])
			f.write('\n')
	print('resulting number of subjects', len(result_sub))
	for i in range(0, len(valid_subjects)):
		if valid_subjects[i] not in result_sub:
			print(valid_subjects[i])


def filter_path():
	# read in path data
	old_path_data = []
	with open(pathfile) as f: 
		for line in f:
			old_path_data.append(json.loads(line))
	# process data
	new_data = []
	for i in range(0, len(old_path_data)): 
		line = old_path_data[i]
		subject = line['subject']
		assignment = line['assignment']
		if subject + ':' + assignment in valid_subjects:
			new_data.append(line)
	# save new data
	result_sub = [] # check the resulting number of subjects
	with open(outfile2, 'w+') as f:
		for i in range(0, len(new_data)): # each line
			json.dump(new_data[i], f)
			if (new_data[i]['subject'] + ':' + new_data[i]['assignment']) not in result_sub:
				result_sub.append(new_data[i]['subject'] + ':' + new_data[i]['assignment'])
			f.write('\n')
	print('resulting number of subjects', len(result_sub))
	for i in range(0, len(valid_subjects)):
		if valid_subjects[i] not in result_sub:
			print(valid_subjects[i])
	return new_data

def sort_by_sub(d):
    '''a helper function for sorting'''
    return d['subject']
def sort_by_ins(d):
    '''a helper function for sorting'''
    return d['instance']

def get_humanlen():
	# get human len for all trials (successful only) from filtered path data
	# load path data
	pathdata = []
	humanlen_list = []
	sub_list = []
	puzzle_list = []
	rt_list = []
	optlen_list = []
	with open(outfile2) as f: 
		for line in f:
			pathdata.append(json.loads(line))
	# sort path data by subject
	pathdata = sorted(pathdata, key=sort_by_sub)
	for i in range(0, len(pathdata)): 
		line = pathdata[i]
		rt = line['rt']
		subject = line['subject']
		assignment = line['assignment']
		subject = subject + ':' + assignment
		complete = line['complete']
		humanlen = line['human_length']
		puzzle = line['instance']
		opt_len = line['optimal_length']
		# ignore surrender and restart and other trivial invalid
		if complete == 'True' and int(humanlen) >= 7:
			rt_list.append(float(rt))
			sub_list.append(subject)
			humanlen_list.append(int(humanlen))
			puzzle_list.append(puzzle)
			optlen_list.append(int(opt_len))
	print('humanlen list len', len(humanlen_list))
	print('sublist len', len(sub_list))
	print('puzzle list len', len(puzzle_list))
	print('rt list len', len(rt_list))
	print('optlen_list', len(optlen_list))
	print(humanlen_list)
	np.save(humanfile + 'humanlen_list.npy', humanlen_list)
	np.save(humanfile + 'sub_list.npy', sub_list)
	np.save(humanfile + 'puzzle_list.npy', puzzle_list)
	np.save(humanfile + 'rt_list.npy', rt_list)
	np.save(humanfile + 'optlen_list.npy', optlen_list)




# filter_move()
filter_path()
# get_humanlen()

# print('\n\n\n')



sys.exit()
#################################### stop ######################################### 
# below is old filtering code, no use




# process move json file
old_move_data = []
# read in data
with open(movefile) as f: # read path data
	for line in f:
		old_move_data.append(json.loads(line))
# sort by subject
old_move_data = sorted(old_move_data, key=sort_by_sub)
new_move_data_tmp = []
new_move_data = [] # outmove
# iterate through line
cur_sub = ''
prev_sub = ''
surrender_error = True
incomplete_error = True
sub_data = []
for i in range(0, len(old_move_data)): # line
	line = old_move_data[i]
	instance = line['instance']
	subject = line['subject']
	move_number = int(line['move_number'])
	cur_sub = subject
	if move_number >= 7:
		surrender_error = False
	if cur_sub != prev_sub and (prev_sub != ''): 
	# begin a new subject, except for the first one
		if (not surrender_error) and (not incomplete_error): # save or not
			new_move_data_tmp.append(sub_data)
		# clean data field for the next subject
		sub_data = []
		surrender_error = True
		incomplete_error = True
	sub_data.append(line)
	if i == len(old_move_data) - 1: # at the end of file
		if (not surrender_error) and (not incomplete_error):
			new_move_data_tmp.append(sub_data)
	prev_sub = subject
# flatten the move file
for sub in new_move_data_tmp:
	for move in new_move_data_tmp:
		new_move_data.append(move)
# sort the new move file by instance 
new_move_data = sorted(new_move_data, key=sort_by_ins)
# save the new moves file
with open(outmove, 'w+') as f:
	for i in range(0, len(new_move_data)): # each line
		json.dump(new_move_data[i], f)
		f.write('\n')


# process new move files for each puzzle
# initialize lists
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	out_file.append(outfile + instance + '_moves_filtered.json')
	out_data.append([])
# read every line of new move data
for line in new_move_data:
	cur_line = json.loads(line)
	cur_ins = cur_line['instance']
	ins_index = all_instances.index(cur_ins)
	# print(ins_index)
	# append current line to corresponding list index
	cur_data = out_data[ins_index]
	cur_data.append(cur_line)
	out_data[ins_index] =  cur_data
# write to file
for i in range(0, len(all_instances)):
    cur_data = out_data[i]
    cur_data = sorted(cur_data, key=sort_by_sub) # sort data by subject
    cur_file = out_file[i]
    cur_file = open(cur_file, 'w+')
    for j in range(0, len(cur_data)):
	    json.dump(cur_data[j], cur_file)
	    cur_file.write('\n')
	    