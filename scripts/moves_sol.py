# generate optimal solutions for each move
# need to run with python365
# each row records the optimla decisions to be made, 
# compare the optimal decisions with the move in the same row for correctness
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Documents/RushHour/exp_data/moves_valid.csv'
out_sol_file = '/Users/chloe/Documents/RushHour/exp_data/moves_sol72.npy'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
header =  ['worker', 'assignment', 'instance', \
			'optlen', 'move_number', 'move', \
			'meta_move', 'rt', 'trial_number']

move_data = pd.read_csv(moves_file)
# print('shape ', move_data.shape)
# print(move_data.loc[0, :])
# logfile = open('/Users/chloe/Documents/RushHour/log.txt', 'w')

out_data = [] # for all moves and trials
my_car_list = None
my_red = None
car_id = None
car_to_pos = None

counter = 0
start_time = time.time()
# for i in range(len(move_data)):
for i in range(0, 72):
	row = move_data.loc[i, :]
	cur_subject = row['worker'] + ':' + row['assignment']
	cur_instance = row['instance']
	move = row['move']
	move_number = row['move_number']
	trial_number = row['trial_number']
	meta_move = row['meta_move']

	if trial_number == 0:
		continue
	if meta_move == 'win' or move == 'r@16':
		out_data.append([])
		continue
	if move_number == 0: # initialize puzzle
		infile = instance_folder + cur_instance + '.json'
		my_car_list, my_red = MAG.json_to_car_list(infile)
		my_board, my_red = MAG.construct_board(my_car_list)
		# optimal solutions at the beginning of each trial
		board_str = MAG.board_to_str(my_board)
		unique_solutions = []
		unique_solutions_set = set()
		# print(board_str)
		sol_list, maxn = solution.main(board_str, 0)
		# print("solution len: ", len(sol_list)) # number of steps in solution
		# print(sol_list)
		sol_list_str = ', '.join(str(sol_list))
		unique_solutions_set.add(sol_list_str)
		unique_solutions.append(sol_list)
		# print("maxn ", maxn)
		for j in range(1, maxn):
			sol_list, _ = solution.main(board_str, j)
			# print("solution len: ", len(sol_list)) # number of steps in solution
			# print(sol_list)
			sol_list_str = ', '.join(str(sol_list))
			if sol_list_str not in unique_solutions_set:
				unique_solutions_set.add(sol_list_str)
				unique_solutions.append(sol_list)
		# print("all optimal solutions ", unique_solutions)
		# append solutions
		out_data.append(unique_solutions)
		# print("len out data ", len(out_data))
		# logfile.write(str(len(out_data)))
		# print('len out_data ', len(out_data))
		# print("iteration count ", i)
		car_id = move[0]
		car_to_pos = int(move[2:])
		continue
	
	# new board
	my_board, my_red = MAG.construct_board(my_car_list)
	# move
	# print("current move: ", car_id, car_to_pos)
	my_car_list, my_red = MAG.move(my_car_list, car_id, car_to_pos)
	my_board, my_red = MAG.construct_board(my_car_list)
	# find solutions
	board_str = MAG.board_to_str(my_board)
	unique_solutions = []
	unique_solutions_set = set()
	print(board_str)
	sol_list, maxn = solution.main(board_str, 0)
	print("solution len: ", len(sol_list)) # number of steps in solution
	# print(sol_list)
	sol_list_str = ', '.join(str(sol_list))
	unique_solutions_set.add(sol_list_str)
	unique_solutions.append(sol_list)
	# print("maxn ", maxn)
	for j in range(1, maxn):
		sol_list, _ = solution.main(board_str, j)
		# print("solution len: ", len(sol_list)) # number of steps in solution
		# print(sol_list)
		sol_list_str = ', '.join(str(sol_list))
		if sol_list_str not in unique_solutions_set:
			unique_solutions_set.add(sol_list_str)
			unique_solutions.append(sol_list)
	print("all unique solutions ", unique_solutions)
	# append solutions
	out_data.append(unique_solutions)
	print("len out data ", len(out_data))
	print("iteration count ", i)
	print("\n\n")
	
	# for next move
	car_id = move[0]
	car_to_pos = int(move[2:])

	# if counter == 26:
		# sys.exit()

	counter += 1
	print('counter', counter)
	if counter % 10 == 0:
		end_time = time.time()
		print("time spent ", end_time - start_time)
		start_time = time.time()

	# end of file
	if counter == len(move_data):
		# new board
		my_board, my_red = MAG.construct_board(my_car_list)
		# move
		my_car_list, my_red = MAG.move(my_car_list, car_id, car_to_pos)
		my_board, my_red = MAG.construct_board(my_car_list)
		# find solutions
		board_str = MAG.board_to_str(my_board)
		unique_solutions = []
		unique_solutions_set = set()
		# print(board_str)
		sol_list, maxn = solution.main(board_str, 0)
		# print("solution len: ", len(sol_list)) # number of steps in solution
		# print(sol_list)
		sol_list_str = ', '.join(str(sol_list))
		unique_solutions_set.add(sol_list_str)
		unique_solutions.append(sol_list)
		# print("maxn ", maxn)
		for j in range(1, maxn):
			sol_list, _ = solution.main(board_str, j)
			# print("solution len: ", len(sol_list)) # number of steps in solution
			# print(sol_list)
			sol_list_str = ', '.join(str(sol_list))
			if sol_list_str not in unique_solutions_set:
				unique_solutions_set.add(sol_list_str)
				unique_solutions.append(sol_list)
		# print("all unique solutions ", unique_solutions)
		# append solutions
		out_data.append(unique_solutions)
		# print("len out data ", len(out_data))
		# print("iteration count ", i)

np.save(out_sol_file, out_data)
# logfile.close()
