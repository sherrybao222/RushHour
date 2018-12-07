# file to list impossible moves in datafile
import json, math
import numpy as np
import MAG, solution
import os

len_file = '/Users/chloe/Documents/RushHour/exp_data/paths.json'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
error_subject = {} # dict storing all subjects that encounter errors
log_output = '/Users/chloe/Documents/RushHour/exp_data/checkdata_log.txt'


f = open(log_output, 'w+')
for i in range(len(all_instances)):
	cur_ins = all_instances[i]
	ins_file = ins_dir + cur_ins + '.json'
	move_file = move_dir + cur_ins + '_moves.json'
	#print('now checking instance ' + str(cur_ins))
	f.write('now checking instance %s\n' % str(cur_ins))

	instance_data = []
	subject_list = [] # list of corresponding subject for successful trials
	trial_data = []
	total_back_move = 0
	is_win = False
	first_subject = True
	print_value_error = True
	# filter successful trials
	with open(move_file) as movefile:
		for line in movefile:
			instance_data.append(json.loads(line))
	for i in range(0, len(instance_data)):  # iterate each line
		line = instance_data[i]
		win = line['meta_move']
		if win == 'win':
			is_win = True
		move_num = line['move_number']
		cur_move = line['move']
		car_to_move = cur_move[0]
		try:
   			move_to_position = int(cur_move[2:])
		except ValueError: # move value error, pass
			if print_value_error: 
				print("Failure w/ value " + cur_move[2:] + ', line' + str(i) + ', instance ' + str(cur_ins))
				print_value_error = False
				f.write("Failure w/ value %s, line %s, instance %s\n" % (cur_move[2:], str(i), str(cur_ins)))
			continue
		if move_num == '0': # begin next trial
			if not is_win and len(trial_data)>0:
				trial_data.pop()
				subject_list.pop()
			trial_data.append([])
			subject_list.append(line['subject'])
			trial_data[-1].append((car_to_move, move_to_position))
			is_win = False
			continue
		trial_data[-1].append((car_to_move, move_to_position))
		if first_subject: # append the subject to the first success trial
			subject_list.append(line['subject'])
			first_subject = False
		if i == len(instance_data) - 1: # the last line 
			if not is_win:
				trial_data.pop()
				subject_list.pop()
	

	print_instance = False # flag if instance has error
	instance_error = 0 # number of error trials in instance
	for i in range(len(trial_data)): # iterate each success trials
		print_trial = False # flag if trial has error
		my_car_list, my_red = MAG.json_to_car_list(ins_file) # initial car list
		my_board, my_red = MAG.construct_board(my_car_list) # initial board
		for j in range(len(trial_data[i])): # process each move
			move = []
			move.append(trial_data[i][j][0])
			move.append(trial_data[i][j][1])
			car_to_move = move[0]
			move_to = int(move[1])
			is_valid, car_found = MAG.check_move(my_car_list, my_board, \
									car_to_move, move_to)
			if is_valid:
				my_car_list, my_red = MAG.move(my_car_list, \
										car_to_move, move_to)
				my_board, my_red = MAG.construct_board(my_car_list)
				new_car_list = MAG.construct_mag(my_board, my_red)
			else: # if current move not valid, pass
				print_trial = True
				# print('################# invalid move at ' + str(trial_data[i][j]) + ', omitted')
				f.write('################# invalid move at %s, omitted\n' % str(trial_data[i][j]))
				continue
		if print_trial:
			# print('error occurs at trial: ' + str(trial_data[i]))
			# print('error occurs with subject: ' + str(subject_list[i]))
			f.write('the above errors occur at trial %s\n' % str(trial_data[i]))
			f.write('the above error trial occurs with subject %s\n' % str(subject_list[i]))
			if not car_found:
				print('car not found at trial: %s' % str(trial_data[i]))
				f.write('car not found at trial: %s' % str(trial_data[i]))
			instance_error += 1
			if subject_list[i] not in error_subject:
				error_subject[subject_list[i]] = 1
			else:
				error_subject[subject_list[i]] += 1
			print_instance = True
	if print_instance:
		print('error instance ' + str(cur_ins) + ' with ' + str(instance_error) + ' erroring trials')
		f.write('error instance %s with %s number of error trials\n' %(str(cur_ins), str(instance_error)))


print('error subject in summary: ')
f.write('error subject in summary: \n')
for e in error_subject: 
	print(str(e), str(error_subject[e]))
	f.write(str(e) + str(error_subject[e]) + '\n')
f.close()


