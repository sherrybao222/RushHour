# BFS model

import MAG
import random, sys, copy
import numpy as np


class Node:
	def __init__(self, cl):
		self.__car_list = cl
		self.__children = []

	def add_child(self, n):
		n.set_parent(self)
		self.__children.append(n)

	def set_parent(self, p):
		self.__parent = p

	def set_value(self, v):
		self.__value = v

	def get_carlist(self):
		return self.__car_list

	def get_red(self):
		for car in self.__car_list:
			if car.tag == 'r':
				return car

	def get_board(self):
		tmp_b, _ = MAG.construct_board(self.__car_list)
		return tmp_b

	def get_value(self):
		self.__value = Value1(self.__car_list, self.get_red())
		return self.__value

	def get_child(self, ind):
		return self.__children[ind]

	def get_children(self):
		return self.__children

	def get_parent(self):
		return self.__parent
	
	def remove_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				c.parent = None
				self.__children.pop(i)
				return	

	def update_carlits(self, cl):
		self.__car_list = cl
		self.__board, self.__red = MAG.construct_board(self.__car_list)
		self.__value = Value1(self.__car_list, self.__red)		

	def print_children(self):
		for i in range(len(self.__children)):
			print(i)
			print('print all children:\n'+MAG.board_to_str(self.__children[i].get_board()))

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




def Value1(car_list2, red):
	'''
	value = w0R * num_cars(red, right)
		+ w1 * num_cars{MAG-level 1} 
		+ w2 * num_cars{MAG-level 2}  
		+ w3 * num_cars{MAG-level 3} 
		+ w4 * num_cars{MAG-level 4} 
		+ w5 * num_cars{MAG-level 5} 
		+ w6 * num_cars{MAG-level 6}
		+ w7 * num_cars{MAG-level 7}  
		+ noise
	'''
	num_parameters = 8
	[w0R, w1, w2, w3, w4, w5, w6, w7] = [-1,-1,-1,-1,-1,-1,-1,-1]
	weights = [w0R, w1, w2, w3, w4, w5, w6, w7]
	noise = np.random.normal(loc=0, scale=1)
	value = 0
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# number of cars at the top level (red only)
	value += w0R * 1 
	# each following level
	for j in range(num_parameters - 1): 
		level = j+1
		new_car_list2 = MAG.clean_level(new_car_list2)
		new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
		cars_from_level = MAG.get_cars_from_level2(new_car_list2, level)
		value += weights[level] * (len(cars_from_level))
	return value+noise


def DropFeatures(delta):
	pass


def Lapse(probability=0.1): 
# return true with a probability
	return random.random() < probability


def Stop(probability=0.3): 
# return true with a probability
	return random.random() < probability


def Determined(root_node): 
# return true if win
	return MAG.check_win(root_node.get_board(), root_node.get_red())


def RandomMove(node):
# make a random move and return the resulted node
	all_moves = MAG.all_legal_moves(node.get_carlist(), node.get_red(), node.get_board())
	(car, pos) = random.choice(all_moves)
	new_list, new_red = MAG.move2(node.get_carlist(), car, pos[0], pos[1])
	new_node = Node(new_list)
	return new_node


def InitializeChildren(root_node):
# initialize the list of children (using all possible moves)
	tmp = copy.deepcopy(root_node)
	all_moves = MAG.all_legal_moves(tmp.get_carlist(), tmp.get_red(), tmp.get_board())

	for i, (tag, pos) in enumerate(all_moves):
		# print('possible move number '+str(i))
		new_list, _ = MAG.move2(tmp.get_carlist(), tag, pos[0], pos[1])
		dummy_child = Node(new_list)
		root_node.add_child(dummy_child)
		
	root_node.print_children()


def SelectNode(root_node):
# return the child with max value
	return ArgmaxChild(root_node)
 

def ExpandNode(node, threshold):
# create all possible nodes under input node, cut the ones below threshold
	InitializeChildren(node)
	Vmax = MaxChildValue(node)
	for child in node.get_children():
		if abs(child.get_value() - Vmax) > threshold:
			node.remove_child(child)


def Backpropagate(this_node, root_node):
	print('backpropagate current node\n'+this_node.board_to_str())
	this_node.set_value(MaxChildValue(this_node))
	if this_node != root_node:
		Backpropagate(this_node.get_parent(), root_node)


def MaxChildValue(node): 
# return the max value from node's children
	Vmax = -float('inf')
	for child in node.get_children():
		if Vmax < child.get_value():
			Vmax = child.get_value()
	return Vmax

def ArgmaxChild(root_node): 
# return the child with max value
	maxChild = None
	for child in root_node.get_children():
		if maxChild == None:
			maxChild = child
		elif maxChild.get_value() < child.get_value():
			maxChild = child
	print('max child selected\n'+maxChild.board_to_str())
	return maxChild

'''
Initial board:
..7756
....56
.RR.5.>
42.333
4211..
.2.00.

'''

def MakeMove(state, delta=0, gamma=0.1, theta=0):
	if Lapse():
		print('Random move made')
		return RandomMove(state)
	else:
		DropFeatures(delta)
		root = state # state is already a node
		print('root node before initialization\n'+root.board_to_str())
		InitializeChildren(root)
		print('root node after initialization\n'+root.board_to_str())
		debug = 0
		while not Stop(gamma) and not Determined(root):
			n = SelectNode(root)
			debug += 1
			print('while iteration number, ' + str(debug))
			print('selected move\n'+n.board_to_str())
			ExpandNode(n, theta)
			print('root after this while iteration:\n'+root.board_to_str())
			Backpropagate(n, root)
			if debug == 1:
				sys.exit()
	return ArgmaxChild(root)


def Main():
	all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	
	# randomly choose a puzzle/instance
	cur_ins = random.choice(all_instances)
	cur_ins = all_instances[0]
	ins_file = ins_dir + cur_ins + '.json'
	
	# construct initial state/node
	my_car_list, my_red = MAG.json_to_car_list(ins_file)
	my_board, my_red = MAG.construct_board(my_car_list)
	# print('Initial board:')
	print(MAG.board_to_str(my_board))
	state_node = Node(my_car_list)
	
	# make a move
	new_node = MakeMove(state_node)
	print('First move made:')
	print(new_node.board_to_str())



if __name__ == '__main__':
    Main()






