from Board import *
from MAG import *
import os, pickle, sys

if __name__ == "__main__":
	datapath = '/Users/yichen/Documents/RushHour/exp_data/data_adopted/'
	outpath = '/Users/yichen/Desktop/preprocessed_positions/'
	all_puzzles = os.listdir(datapath)
	for puzzle in all_puzzles:
		if not puzzle.endswith('.json'):
			continue
		puzzle = puzzle[:-5]
		print(puzzle)
		# create puzzle directory
		if not os.path.exists(outpath+puzzle+'/'):
			os.mkdir(outpath+puzzle+'/')	
		else:
			continue
		# create puzzle info file and pickle it
		create_puzzle_info_file(puzzle)
		# create initial board
		board = json_to_board(datapath+puzzle+'.json')
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
			pickle.dump(b_dict, open(outpath+puzzle+'/'+position_id+'.p', 'wb'))
			# print this position board for debug
			# id_to_board(position_id, puzzle).print_board()
			# b.print_board()
		# report number of legal positions
		print(len(list(all_legal_positions)))
		# break