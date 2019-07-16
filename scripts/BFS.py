# BFS model
# py27
import MAG
import random, sys, copy
import numpy as np


class Node:
	car_list = None
	red = None
	board = None
	value = 0
	children = []
	parent = None

	def __init__(self, cl):
		self.car_list = cl
		self.board, self.red = MAG.construct_board(self.car_list)
		self.value = Value1(self.car_list, self.red)

	def add_child(self, n):
		# c.parent = self
		self.children.append(n)

	def set_parent(self, p):
		self.parent = p

	def remove_child(self, c):
		for i in range(len(self.children)):
			if self.children[i] == c:
				c.parent = None
				self.children.pop(i)
				return		

	def update_carlits(self, cl):
		self.car_list = cl
		self.board, self.red = MAG.construct_board(self.car_list)
		self.value = Value1(self.car_list, self.red)		

	def print_children(self):
		for i in range(len(self.children)):
			print(i)
			print('print child:\n'+MAG.board_to_str(self.children[i].board))

	def print_children2(self):
		for i in range(0, len(self.children)):
			print('print child: \n'+self.children[i])



def Value1(car_list, red):
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
	my_board, my_red = MAG.construct_board(car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	# number of cars at the top level (red only)
	value += w0R * 1 
	# each following level
	for j in range(num_parameters - 1): 
		level = j+1
		new_car_list = MAG.clean_level(new_car_list)
		new_car_list = MAG.assign_level(new_car_list, my_red)
		cars_from_level = MAG.get_cars_from_level2(new_car_list, level)
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
	return MAG.check_win(root_node.board, root_node.red)


def RandomMove(node):
# make a random move and return the resulted node
	all_moves = MAG.all_legal_moves(node.car_list, node.red, node.board)
	(car, pos) = random.choice(all_moves)
	new_list, new_red = MAG.move2(node.car_list, car.tag, pos[0], pos[1])
	new_node = Node(new_list)
	return new_node


def InitializeChildren(root_node):
# initialize the list of children (using all possible moves)
	tmp = copy.deepcopy(root_node.car_list)
	print('root before initialization:\n'+MAG.board_to_str(root_node.board))
	all_moves = MAG.all_legal_moves(root_node.car_list, root_node.red, root_node.board)
	for (tag, pos) in all_moves:
		new_list, _ = MAG.move2(tmp, tag, pos[0], pos[1])
		# copy_new_list = copy.deepcopy(new_list)
		# child = Node(copy_new_list)
		# child.set_parent(root_node)
		# root_node.add_child(child)
		# child.parent = root_node
		root_node.children.append(Node(copy.deepcopy(new_list)))
		# root_node.children.append(tag)
		print('child: \n'+MAG.board_to_str(Node(new_list).board))
		# print('child: \n'+MAG.board_to_str(child.board))
	# root_node.add_child(Node(tmp))
	# root_node.remove_child(Node(tmp))
	root_node.print_children()
	# root_node.print_children2()
	print('root after initialization\n'+MAG.board_to_str(root_node.board))
	

def SelectNode(root_node):
# return the child with max value
	return ArgmaxChild(root_node)
 

def ExpandNode(node, threshold):
# create all possible nodes under input node, cut the ones below threshold
	InitializeChildren(node)
	Vmax = MaxChildValue(node)
	for child in node.children:
		if abs(child.value - Vmax) > threshold:
			node.remove_child(child)


def Backpropagate(this_node, root_node):
	print('backpropagate\n'+MAG.board_to_str(this_node.board))
	this_node.value = MaxChildValue(this_node)
	if this_node != root_node:
		Backpropagate(this_node.parent, root_node)


def MaxChildValue(node): 
# return the max value from node's children
	Vmax = -float('inf')
	for child in node.children:
		if Vmax < child.value:
			Vmax = child.value
	return Vmax

def ArgmaxChild(root_node): 
# return the child with max value
	maxChild = None
	for child in root_node.children:
		print('child\n'+MAG.board_to_str(child.board))
		if maxChild == None:
			maxChild = child
		elif maxChild.value < child.value:
			maxChild = child
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
		InitializeChildren(root)
		sys.exit()
		debug = 0
		while not Stop(gamma) and not Determined(root):
			print('--------root--------\n'+MAG.board_to_str(root.board))
			n = SelectNode(root)
			debug += 1
			print(debug)
			print('selected node\n'+MAG.board_to_str(n.board))
			ExpandNode(n, theta)
			print('root:\n'+MAG.board_to_str(root.board))
			Backpropagate(n, root)
			# print('root:\n'+MAG.board_to_str(root.board))
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
	print('Initial board:')
	print(MAG.board_to_str(my_board))
	state_node = Node(my_car_list)
	
	# make a move
	new_node = MakeMove(state_node)
	print('Move made:')
	print(MAG.board_to_str(new_node.car_list))



if __name__ == '__main__':
    Main()






