# BFS model

import MAG
import random, sys, copy
import numpy as np


class Node:
	def __init__(self, cl):
		self.__car_list = cl
		self.__children = []

	def add_child(self, c):
		self.__children.append(c)

	def get_child(self, ind):
		return self.__children[ind]

	def get_carlist(self):
		return self.__car_list

	def get_red(self):
		for car in self.__car_list:
			if car.tag == 'r':
				return car

	def get_board(self):
		tmp_b, _ = MAG.construct_board(self.__car_list)
		return tmp_b

	def board_to_str(self):
		tmp_board, tmp_red = MAG.construct_board(self.__car_list)
		out_str = ''
		for i in range(tmp_board.height):
			for j in range(tmp_board.width):
				cur_car = tmp_board.board_dict[(j, i)]
				if cur_car == None:
					out_str += '.'
					if i == 2 and j == 5:
						out_str += '>'
					continue
				if cur_car.tag == 'r':
					out_str += 'R'
				else:
					out_str += cur_car.tag
				if i == 2 and j == 5:
					out_str += '>'
			out_str += '\n'
		return out_str


def InitializeChildren(root_node):

	tmp = copy.deepcopy(root_node)
	all_moves = MAG.all_legal_moves(tmp.get_carlist(), tmp.get_red(), tmp.get_board())
	count = 0
	
	for i, (tag, pos) in enumerate(all_moves):
		print(i)
		new_list, _ = MAG.move2(tmp.get_carlist(), tag, pos[0], pos[1])
		dummy_child = Node(new_list)
		root_node.add_child(dummy_child)

		print(root_node.get_child(0).board_to_str())
		count += 1
		if count == 5:
			sys.exit()



def MakeMove(state, delta=0, gamma=0.1, theta=0):
	root = state # state is already a node
	InitializeChildren(root)
	return

def Main():
	all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'

	cur_ins = all_instances[0]
	ins_file = ins_dir + cur_ins + '.json'
	
	# construct initial state/node
	init_car_list, init_red = MAG.json_to_car_list(ins_file)
	init_board, init_red = MAG.construct_board(init_car_list)
	print('Initial board:')
	print(MAG.board_to_str(init_board))
	state_node = Node(init_car_list)
	
	# make a move
	new_node = MakeMove(state_node)
	print('Move made:')
	print(MAG.board_to_str(new_node.get_carlist()))



if __name__ == '__main__':
    Main()






