from Board import *
from MAG import *
import os, pickle, sys

if __name__ == "__main__":
	datapath = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	outpath = '/Users/chloe/Documents/RushHour/exp_data/preprocessed_positions/'
	all_instances = os.listdir(datapath)
	for instance in all_instances:
		instance = instance[:-5]
		print(instance)
		# create puzzle directory
		if not os.path.exists(outpath+instance+'/'):
			os.mkdir(outpath+instance+'/')	
		# create puzzle info file and pickle it
		create_puzzle_info_file(instance)
		# create initial board
		board = json_to_board(datapath+instance+'.json')
		# find all legal positions
		all_legal_positions = generate_all_positions_of_board(board)
		for position in all_legal_positions: # for each legal position, 
			# make position board and id
			b = position_to_board(position)
			position_id = make_id(b)
			# initialize position dictionary
			b_dict = {'children_ids':[], 'children_boards':[], 'children_mags':[]}
			# find all legal children of this position
			children_boards = all_legal_moves(b)
			# save child board and child mag, and child id
			for child in children_boards:
				identity = make_id(child)
				b_dict['children_ids'].append(identity)
				b_dict['children_boards'].append(child)
				mag = MAG(child)
				mag.construct()
				b_dict['children_mags'].append(mag)
			# pickle the dictioary and name by position id
			pickle.dump(b_dict, open(outpath+instance+'/'+position_id+'.p', 'wb'))
		# report number of legal positions
		print(len(list(all_legal_positions)))
		break