# BFS model

import MAG
import random, sys, copy
import numpy as np
import matplotlib.pyplot as plt

# global w0, w1, w2, w3, w4, w5, w6, w7, noise, weights
# w0 = None
# w1 = None
# w2 = None
# w3 = None
# w4 = None
# w5 = None
# w6 = None
# w7 = None
# noise = None
# weights = [w0, w1, w2, w3, w4, w5, w6, w7]
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']	

class Node:
	def __init__(self, cl):
		self.__car_list = cl
		self.__children = []
		self.__value = None
		self.__value_level = None

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
		if self.__value == None:
			self.__value, self.__value_level = Value1(self.__car_list, self.get_red())
		return self.__value

	def get_value_level(self):
		return self.__value_level

	def get_child(self, ind):
		return self.__children[ind]

	def get_children(self):
		return self.__children

	def find_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				return i
		return None

	def clean_children(self):
		self.__children = []

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
		self.__value, self.__value_level = Value1(self.__car_list, self.__red)		

	def print_children(self):
		print('total number of children: '+str(len(self.__children)))
		for i in range(2):
			print('print first two children: '+str(i))
			print(MAG.board_to_str(self.__children[i].get_board()))

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
	value = w0 * num_cars{MAG-level 0}
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
	value_level = []
	# [w0, w1, w2, w3, w4, w5, w6, w7] = [-1,-1,-1,-1,-1,-1,-1,-1]
	# weights = [w0, w1, w2, w3, w4, w5, w6, w7]
	# noise = np.random.normal(loc=0, scale=5)
	value = 0
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# number of cars at the top level (red only)
	value += w0 * 1 
	value_level.append(1)
	# each following level
	for j in range(num_parameters - 1): 
		level = j+1
		new_car_list2 = MAG.clean_level(new_car_list2)
		new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
		cars_from_level = MAG.get_cars_from_level2(new_car_list2, level)
		value += weights[level] * (len(cars_from_level))
		value_level.append(len(cars_from_level))
	return value+noise, value_level


def DropFeatures(delta):
	pass


def Lapse(probability=0.1): 
	''' return true with a probability '''
	return random.random() < probability


def Stop(probability=0.3): 
	''' return true with a probability '''
	return random.random() < probability


def Determined(root_node): 
	''' return true if win, false otherwise '''
	return MAG.check_win(root_node.get_board(), root_node.get_red())


def RandomMove(node):
	''' make a random move and return the resulted node '''
	# print('Random move made')
	return random.choice(node.get_children())
	

def InitializeChildren(root_node):
	''' initialize the list of children (using all possible moves) '''
	if len(root_node.get_children()) == 0:
		tmp = copy.deepcopy(root_node)
		all_moves = MAG.all_legal_moves(tmp.get_carlist(), tmp.get_red(), tmp.get_board())
		for i, (tag, pos) in enumerate(all_moves):
			new_list, _ = MAG.move2(tmp.get_carlist(), tag, pos[0], pos[1])
			dummy_child = Node(new_list)
			root_node.add_child(dummy_child)
	# root_node.print_children()


def SelectNode(root_node):
	''' return the child with max value '''
	return ArgmaxChild(root_node)
 

def ExpandNode(node, threshold):
	''' create all possible nodes under input node, 
	cut the ones below threshold '''
	if len(node.get_children()) == 0:
		InitializeChildren(node)
	Vmax = MaxChildValue(node)
	for child in node.get_children():
		if abs(child.get_value() - Vmax) > threshold:
			node.remove_child(child)


def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	# print('backpropagate current node\n'+this_node.board_to_str())
	this_node.set_value(MaxChildValue(this_node))
	if this_node != root_node:
		Backpropagate(this_node.get_parent(), root_node)


def MaxChildValue(node): 
	''' return the max value from node's children '''
	Vmax = -float('inf')
	for child in node.get_children():
		if Vmax < child.get_value():
			Vmax = child.get_value()
	return Vmax


def ArgmaxChild(root_node): 
	''' return the child with max value '''
	maxChild = None
	for child in root_node.get_children():
		if maxChild == None:
			maxChild = child
		elif maxChild.get_value() < child.get_value():
			maxChild = child
	# print('max child selected\n'+maxChild.board_to_str())
	return maxChild


def MakeMove(state, delta=0, gamma=0.1, theta=float('inf')):
	''' returns an optimal move to make next, given current state '''
	root = state # state is already a node
	# print('root node before initialization\n'+root.board_to_str())
	InitializeChildren(root)
	# print('root node after initialization\n'+root.board_to_str())
	if Lapse():
		# print('Random move made')
		return RandomMove(root)
	else:
		DropFeatures(delta)
		debug = 0
		while not Stop(gamma) and not Determined(root):
			n = SelectNode(root)
			debug += 1
			# print('while iteration number, ' + str(debug))
			# print('selected move\n'+n.board_to_str())
			ExpandNode(n, theta)
			Backpropagate(n, root)
	# print('selected move\n'+ArgmaxChild(root).board_to_str())
	return ArgmaxChild(root)


def EstimateProb(root_node, sol_str='', iteration=1000):
	''' Estimate the probability of next possible moves given the root node '''
	InitializeChildren(root_node)
	frequency = [0] * len(root_node.get_children())
	for i in range(iteration):
		new_node = MakeMove(root_node)
		child_idx = root_node.find_child(new_node)
		# print(child_idx)
		frequency[child_idx] += 1
	
	frequency = np.array(frequency, dtype=np.float32)/iteration
	chance = float(1)/float(len(frequency))

	sol_idx = None
	sol_value_level = None
	for i in range(len(root_node.get_children())):
		# print('number of children '+str(len(root_node.get_children())))
		# print('\nChild Board '+str(i)+'\n'+root_node.get_child(i).board_to_str())
		if root_node.get_child(i).board_to_str() == sol_str:
			# print('sol_idx='+str(i))
			# print('sol_prob='+str(frequency[i]))
			sol_idx = i
			sol_value_level = root_node.get_child(i).get_value_level()

	return root_node.get_children(), frequency, sol_idx, sol_value_level, chance


def SolutionProb(instance):
	''' print probability of selecting solution moves 
		at each position along the optimal solution '''
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	solution = np.load('/Users/chloe/Documents/RushHour/exp_data/' + instance + '_solution.npy')
	print('instance '+instance)
	print('optimal solution length '+str(len(solution)))
	sol_prob = []
	chance = []
	sol_vlevel = []

	# construct initial state/node
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	print('Initial board:\n'+MAG.board_to_str(initial_board))
	initial_node = Node(initial_car_list)

	cur_node = initial_node
	cur_carlist = initial_car_list
	
	for i in range(len(solution)):
		# print('\n---------- move number '+str(i+1)+' -----------')
		# print('current board\n'+cur_node.board_to_str())		
		sol = solution[i]
		car_to_move = sol[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(sol[2:])
		cur_carlist, _ = MAG.move_by(cur_carlist, car_to_move, move_by)
		if i == len(solution)-1:
			cur_carlist, _ = MAG.move2(cur_carlist, car_to_move, 4, 2)
		cur_board, _ = MAG.construct_board(cur_carlist)
		sol_board_str = MAG.board_to_str(cur_board)
		# print('solution child\n'+sol_board_str)
		_, frequency, sol_idx, v_level, ch= EstimateProb(cur_node, sol_str=sol_board_str)
		cur_node = cur_node.get_child(sol_idx)
		sol_prob.append(frequency[sol_idx])
		sol_vlevel.append(v_level)
		chance.append(ch)

	return sol_prob, sol_vlevel, chance
	


def Main(this_w0=-1, this_w1=-1, this_w2=-1, this_w3=-1, this_w4=-1, this_w5=-1, this_w6=-1, this_w7=-1, this_noise=np.random.normal(loc=0, scale=1)):
	# randomly choose a puzzle/instance
	# cur_ins = random.choice(all_instances)
	cur_ins = all_instances[-1]
	global w0, w1, w2, w3, w4, w5, w6, w7, noise, weights
	w0 = this_w0
	w1 = this_w1
	w2 = this_w2
	w3 = this_w3
	w4 = this_w4
	w5 = this_w5
	w6 = this_w6
	w7 = this_w7
	noise = this_noise
	weights = [w0, w1, w2, w3, w4, w5, w6, w7]

	# print(type(w0))
	return SolutionProb(cur_ins)


def PlotAllSol(this_w0=-1, this_w1=-5, this_w2=-4, this_w3=-3, this_w4=-1, this_w5=-1, this_w6=-1, this_w7=-1, this_noise=np.random.normal(loc=0, scale=1)):
	# randomly choose a puzzle/instance
	# cur_ins = random.choice(all_instances)
	
	global w0, w1, w2, w3, w4, w5, w6, w7, noise, weights
	w0 = this_w0
	w1 = this_w1
	w2 = this_w2
	w3 = this_w3
	w4 = this_w4
	w5 = this_w5
	w6 = this_w6
	w7 = this_w7
	noise = this_noise
	weights = [w0, w1, w2, w3, w4, w5, w6, w7]

	dummy = np.arange(15)
	all_likelihood = []
	all_chance = []

	for i in range(len(all_instances)):
		cur_ins = all_instances[i]
		likelihood, vlevel, chance = SolutionProb(cur_ins)
		# print(likelihood)
		# plt.plot(likelihood)
		all_likelihood.append(likelihood)

	# plt.plot([0.05]*15, '-.', color="#333333")
	# plt.show()
	return all_likelihood

	

# PlotAllSol()

# if __name__ == '__main__':
#     Main()






