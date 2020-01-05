''' 
BFS model self-defined ll function,
speeded version, prepared for BADS in MATLAB,
now working but very slow (generating data every time).
py27
'''

import MAG, time
import random, sys, copy, os, pickle
import numpy as np
from operator import methodcaller
import multiprocessing as mp
from numpy import recfromcsv
from json import dump, load

class Params:
	def __init__(self, w1, w2, w3, w4, w5, w6, w7):
		self.w0 = 0
		self.w1 = w1
		self.w2 = w2
		self.w3 = w3
		self.w4 = w4
		self.w5 = w5
		self.w6 = w6
		self.w7 = w7
		self.weights = [w0, w1, w2, w3, w4, w5, w6, w7]
		self.num_weights = len(self.weights)
		self.mu = 0
		self.sigma = 1

class Node:
	def __init__(self, cl):
		self.__car_list = cl
		self.__children = []
		self.__value = None
		self.__red = None
		self.__board = None #str
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
		if self.__red == None:
			for car in self.__car_list:
				if car.tag == 'r':
					self.__red = car
		return self.__red
	def get_board(self):
		tmp_b, _ = construct_board(self.__car_list)
		return tmp_b
	def get_value(self):
		if self.__value == None:
			self.__value = self.heuristic_value_function()
		return self.__value
	def get_child(self, ind):
		return self.__children[ind]
	def get_children(self):
		return self.__children
	def find_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				return i
		return None
	def find_child_by_str(self, bstr):
		for i in range(len(self.__children)):
			if self.__children[i].board_to_str() == bstr:
				return i
		return None
	def get_parent(self):
		return self.__parent
	def remove_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				c.parent = None
				self.__children.pop(i)
				return	
	def board_to_str(self):
		if self.__board == None:
			tmp_board, tmp_red = construct_board(self.__car_list)
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
			self.__board = out_str
		return self.__board
	def heuristic_value_function(self):
		'''
		value = w0 * num_cars{MAG-level RED}
			+ w1 * num_cars{MAG-level 1} 
			+ w2 * num_cars{MAG-level 2}  
			+ w3 * num_cars{MAG-level 3} 
			+ w4 * num_cars{MAG-level 4} 
			+ w5 * num_cars{MAG-level 5} 
			+ w6 * num_cars{MAG-level 6}
			+ w7 * num_cars{MAG-level 7}  
			+ noise
		'''
		noise = np.random.normal(loc=params.mu, scale=params.sigma)
		# initialize MAG
		my_board2, my_red2 = construct_board(self.__car_list)
		new_car_list2 = construct_mag(my_board2, my_red2)
		# each following level
		new_car_list2 = assign_level(new_car_list2)
		value = np.sum(np.array(get_num_cars_from_levels(new_car_list2, params.num_weights-1)) * np.array(params.weights))
		return value+noise


######################################## MAG CONTENT ##############

class Car:
	start, end, length = [0,0], [0,0], 0 # [hor,ver]
	tag = ''
	puzzle_tag = ''
	orientation = ''
	edge_to = []
	visited = False
	level = []
	def __init__(self, s, l, t, o, p):
		self.start = s
		self.length = l
		self.tag = t
		self.orientation = o
		self.puzzle_tag = p
		if self.orientation == 'horizontal':
			self.end = [self.start[0] + self.length - 1, self.start[1]]
		elif self.orientation == 'vertical':
			self.end = [self.start[0], self.start[1] + self.length - 1]
		self.edge_to = []
		self.visited = False

class Board:
	height, width = 6, 6
	board_dict = {}
	puzzle_tag = ''
	def __init__(self):
		for i in range(0, self.height):
			for j in range(0, self.width):
				self.board_dict[(j, i)] = None

def json_to_car_list(filename):
	with open(filename,'r') as data_file:
		car_list = []
		data = load(data_file)
		red = ''
		for c in data['cars']:
			cur_car = Car(s = [int(c['position'])%6, int(c['position']/6)], \
				l = int(c['length']), t = c['id'], o = c['orientation'], p = data['id'])
			car_list.append(cur_car)
			if cur_car.tag == 'r':
				red = cur_car
	return car_list, red

def move(car_list, car_tag, to_position): 
	'''
		make a move and return the new car list, 
		single position label
	'''
	new_list2 = []
	red = ''
	for i in range(len(car_list)):
		cur_car = car_list[i]
		if cur_car.tag == car_tag:
			new_car = Car(s = [int(to_position)%6, int(to_position/6)],\
				l = int(cur_car.length), t = car_tag, \
				o = cur_car.orientation, p = cur_car.puzzle_tag)
		else:
			new_car = Car(s = [int(cur_car.start[0]), int(cur_car.start[1])], \
							l = int(cur_car.length), t = cur_car.tag,\
							o = cur_car.orientation, p = cur_car.puzzle_tag)
		new_list2.append(new_car)
		if new_list2[i].tag == 'r':
			red = new_list2[i]
	return new_list2, red

def move2(car_list, car_tag, to_position1, to_position2): 
	'''
	make a move and return the new car list, x and y
	'''
	new_list2 = []
	red = ''
	for i in range(len(car_list)):
		cur_car = car_list[i]
		if cur_car.tag == car_tag:
			new_car = Car(s = [int(to_position1), int(to_position2)],\
				l = int(cur_car.length), t = car_tag, \
				o = cur_car.orientation, p = cur_car.puzzle_tag)
		else:
			new_car = Car(s = [int(cur_car.start[0]), int(cur_car.start[1])], \
							l = int(cur_car.length), t = cur_car.tag,\
							o = cur_car.orientation, p = cur_car.puzzle_tag)
		new_list2.append(new_car)
		if new_list2[i].tag == 'r':
			red = new_list2[i]
	return new_list2, red

def construct_board(car_list):
	board = Board()
	red = ''
	for car in car_list:
		if car.tag == 'r':
			red = car
		cur_start = car.start
		cur_len = car.length
		cur_orientation = car.orientation
		occupied_space = []
		if cur_orientation == 'horizontal':
			for i in range(0, cur_len):
				occupied_space.append((cur_start[0] + i, cur_start[1]))
		elif cur_orientation == 'vertical':
			for i in range(0, cur_len):
				occupied_space.append((cur_start[0], cur_start[1] + i))
		for j in range(0, len(occupied_space)):
			board.board_dict[occupied_space[j]] = car
	return board, red

def all_legal_moves(car_list, board):
	moves = []
	for i in range(len(car_list)):
		cur_car = car_list[i]
		if cur_car.orientation == 'horizontal':
			cur_position1 = cur_car.start[0] - 1 # search left
			cur_position2 = cur_car.start[1]
			while(cur_position1 >= 0 and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1, cur_position2]))
				cur_position1 -= 1
			cur_position1 = cur_car.end[0] + 1 # search right
			while(cur_position1 < board.width and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1-cur_car.length+1, cur_position2]))
				cur_position1 += 1
		if cur_car.orientation == 'vertical':
			cur_position1 = cur_car.start[0]
			cur_position2 = cur_car.start[1] - 1 # search up
			while(cur_position2 >= 0 and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1, cur_position2]))
				cur_position2 -= 1
			cur_position2 = cur_car.end[1] + 1 # searc down
			while(cur_position2 < board.height and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1, cur_position2-cur_car.length+1]))
				cur_position2 += 1
	return moves
			
def check_win(board, red): # return true if current board state can win
	cur_position = red.end[0] + 1 # search right of red car
	while(cur_position < board.width):
		if board.board_dict[(cur_position, red.start[1])] is not None:
			return False
		cur_position += 1
	return True

def assign_level(car_list): 
	''' 
	assign level to each car
	'''
	red = None
	for car in car_list: # clean levels of each car 
		car.level = []
		if car.tag == 'r': # find red
			red = car
	queue = []
	visited = []
	red.level.append(0)
	queue.append(red)
	visited.append(red)
	while queue: # bfs, assign levels
		cur_car = queue.pop(0)
		for child in cur_car.edge_to:
			child.level.append(cur_car.level[-1] + 1)
			if child not in visited:
				queue.append(child)
				visited.append(child)
	return visited

def get_num_cars_from_levels(car_list, highest_level):
	''' 
		return the number of cars at each level 
		highest_level >= any possible min(cur_car.level)
	'''
	list_toreturn = [0] * (highest_level+1)
	for cur_car in car_list:
		if cur_car.level != []:
			list_toreturn[min(cur_car.level)] += 1
	return list_toreturn

def construct_mag(board, red):
	'''
		assign graph children, return a new car list
	'''
	queue = []
	finished_list = set() # list to be returned, all cars in MAG
	i = board.width - 1
	for i in range(red.end[0], board.width): # obstacles in front of red, include red
		cur_car = board.board_dict[(i, red.end[1])]
		if cur_car is not None: #exists on board
			queue.append(cur_car)
			if cur_car.tag != 'r':
				red.edge_to.append(cur_car)
				cur_car.visited = True
	red.visited = True
	while len(queue) != 0: # obstacle blockers
		cur_car	= queue.pop()
		if cur_car.tag == 'r':
			continue
		if cur_car.orientation == 'vertical': # vertical
			if cur_car.start[1] > 0:
				j = cur_car.start[1] - 1
				while j >= 0: # upper
					meet_car =  board.board_dict[(cur_car.start[0], j)]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					j -= 1
			if cur_car.end[1] < board.height - 1:
				k = cur_car.end[1] + 1
				while k <= board.height - 1: # lower
					meet_car =  board.board_dict[(cur_car.start[0], k)]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					k += 1
		elif cur_car.orientation == 'horizontal': # or horizontal
			if cur_car.start[0] > 0:
				j = cur_car.start[0] - 1
				while j >= 0: # left
					meet_car =  board.board_dict[(j, cur_car.start[1])]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					j -= 1
			if cur_car.end[0] < board.width - 1:
				k = cur_car.end[0] + 1
				while k <= board.width - 1: # right
					meet_car =  board.board_dict[(k, cur_car.start[1])]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					k += 1
		cur_car.visited = True # mark
	# clean all visited flags
	for i in range(0, board.height):
		for j in range(0, board.width):
			cur_car = board.board_dict[(i, j)]
			if cur_car is not None:
				cur_car.visited = False
				finished_list.add(cur_car)
	return list(finished_list)
	


if __name__ == '__main__':
	trial_start = 21 # starting row number in the raw data
	trial_end = 24 # inclusive
	sub_data = recfromcsv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	cur_carlist = None
	cur_node = None
	for i in range(trial_start-2, trial_end-1): 
		print('Line number '+str(i))
		row = sub_data[i]
		if row['event'] == 'start':
			instance = row['instance']
			ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
			initial_car_list, _ = json_to_car_list(ins_file)
			cur_carlist = initial_car_list
			cur_node = Node(cur_carlist)
			print('Instance '+str(instance))
			print('Initial board \n'+cur_node.board_to_str())
			print('----------------------------------')
			continue
		cur_node = Node(cur_carlist)
		print('Current Board:\n'+cur_node.board_to_str())
		print('Car Levels: ')
		cur_board, cur_red = construct_board(cur_carlist)
		cur_carlist = construct_mag(cur_board, cur_red)
		assign_level(cur_carlist)
		for car in cur_carlist:
			print('\tCar tag '+car.tag+', level '+str(car.level))
		print('Number of cars at each level: '+str(get_num_cars_from_levels(cur_carlist, 7)))
		print('\nLegal Moves:')
		for (tag, pos) in all_legal_moves(cur_carlist, construct_board(cur_carlist)[0]):
			print(Node(move2(cur_carlist, tag, pos[0], pos[1])[0]).board_to_str())
		print('----------------------------------')
		# make human move
		piece = row['piece']
		move_to = int(row['move'])
		cur_carlist, _ = move(cur_carlist, piece, move_to) 



######################################## BFS MODEL ##############

def DropFeatures(delta):
	pass

def Lapse(probability=0.05): 
	''' return true with a probability '''
	return random.random() < probability

def Stop(probability=0.05): 
	''' return true with a probability '''
	return random.random() < probability

def Determined(root_node): 
	''' return true if win, false otherwise '''
	return check_win(root_node.get_board(), root_node.get_red())

def RandomMove(node):
	''' make a random move and return the resulted node '''
	return random.choice(node.get_children())
	
def InitializeChildren(root_node):
	''' initialize the list of children (using all possible moves) '''
	if len(root_node.get_children()) == 0:
		all_moves = all_legal_moves(root_node.get_carlist(), root_node.get_board())
		root_car_list = root_node.get_carlist()
		for (tag, pos) in all_moves:
			new_list, _ = move2(root_car_list, tag, pos[0], pos[1])
			dummy_child = Node(new_list)
			root_node.add_child(dummy_child)

def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	while len(n.get_children()) != 0:
		n = ArgmaxChild(n)
	return n, []
 
def ExpandNode(node, threshold):
	''' create all possible nodes under input node, 
	cut the ones below threshold '''
	Vmaxchild = ArgmaxChild(node)
	Vmax = Vmaxchild.get_value()
	for child in node.get_children():
		if abs(child.get_value() - Vmax) > threshold:
			node.remove_child(child)
	return Vmaxchild

def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	this_node.set_value(ArgmaxChild(this_node).get_value())
	if this_node != root_node:
		Backpropagate(this_node.get_parent(), root_node)

def ArgmaxChild(root_node): 
	''' return the child with max value '''
	return max(root_node.get_children(), key=methodcaller('get_value'))
	
def MakeMove(state, delta=0, gamma=0.05, theta=10):
	''' returns an optimal move to make next, given current state '''
	root = state # state is already a node
	InitializeChildren(root)
	if Lapse():
		return RandomMove(root), [], []
	else:
		DropFeatures(delta)
		considered_node2 = [] # new node expanded along the branch in this iteration
		while not Stop(probability=gamma) and not Determined(root):
			n, traversed = SelectNode(root)
			n2 = ExpandNode(n, theta)
			considered_node2.append(n2)
			Backpropagate(n, root)
			if n2.get_value() >= abs(np.random.normal(loc=params.mu, scale=params.sigma)): # terminate the algorithm if found a terminal node
				break
	return ArgmaxChild(root), [], considered_node2

def ibs(root_node, expected_board=''):
	''' 
		inverse binomial sampling: 
		return the number of simulations until hit target
	'''
	InitializeChildren(root_node)
	num_simulated = 0
	hit = False
	while not hit:
		new_node, _, _ = MakeMove(root_node)
		num_simulated += 1
		if new_node.board_to_str() == expected_board:
			hit = True
	return num_simulated

def harmonic_sum(n):
	''' return sum of harmonic series from 1 to k '''
	i = 1
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s

def my_ll_parallel(w1, w2, w3, w4, w5, w6, w7): # parallel computing
	start_time = time.time()
	params = Params(w1, w2, w3, w4, w5, w6, w7)
	with open("/Users/chloe/Documents/RushHour/scripts/node_list_03.pickle", "r") as fp:
		node_list = pickle.load(fp)
	with open("/Users/chloe/Documents/RushHour/scripts/expected_list_03.pickle", "r") as fp:
		user_choice = pickle.load(fp)
	pool = mp.Pool(processes=mp.cpu_count())
	all_ibs_obj = [pool.apply_async(ibs, args=(cur, exp_str)) for cur, exp_str in zip(node_list, user_choice)]
	all_ibs_result = [r.get() for r in all_ibs_obj]
	pool.close()
	pool.join()
	all_ll = []
	for n in all_ibs_result:
		all_ll.append(harmonic_sum(n))
	print('sampl size '+str(len(all_ll)))
	ll_result = -np.sum(all_ll)
	print('time '+str(time.time() - start_time))
	return ll_result

def create_data():
	trial_start = 2 # 2072 # starting row number in the raw data
	trial_end = 65 # 2114
	sub_data = recfromcsv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	# construct initial node
	node_list = [] # list of node from data
	user_choice = [] # list of expected human move node, str
	cur_node = None
	cur_carlist = None
	for i in range(trial_start-2, trial_end-1):
		# load data from datafile
		row = sub_data[i]
		if row['event'] == 'start':
			instance = row['instance']
			ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
			initial_car_list, _ = json_to_car_list(ins_file)
			initial_node = Node(initial_car_list)
			cur_node = initial_node
			cur_carlist = initial_car_list
			continue
		piece = row['piece']
		move_to = int(row['move'])
		node_list.append(cur_node) # previous board position
		# create human move
		cur_carlist, _ = move(cur_carlist, piece, move_to)
		cur_node = Node(cur_carlist)
		user_choice.append(cur_node.board_to_str())
	# save list
	with open("/Users/chloe/Documents/RushHour/scripts/node_list_03.pickle", "w") as fp:
		pickle.dump(node_list, fp)
	with open("/Users/chloe/Documents/RushHour/scripts/user_choice_03.pickle", "w") as fp:
		pickle.dump(user_choice, fp)

def estimate_prob(root_node, expected_board='', iteration=100):
	''' 
		Estimate the probability of next possible moves 
		given the root node 
	'''
	first_iteration = True
	frequency = None
	sol_idx = None
	
	for i in range(iteration):
		new_node, _, _ = MakeMove(root_node)
		if first_iteration:
			frequency = [0] * len(root_node.get_children())
			first_iteration = False
		child_idx = root_node.find_child(new_node)
		frequency[child_idx] += 1
	# turn frequency into probability
	frequency = np.array(frequency, dtype=np.float32)/iteration 
	for i in range(len(root_node.get_children())):
		if root_node.get_child(i).board_to_str() == expected_board:
			sol_idx = i
	return root_node.get_children(), frequency, sol_idx, [], []


# if __name__ == '__main__':
	# my_ll_parallel(-0.5391, -3.9844, 3.8281, -5.5469, -2.5781, -8.8867, -2.5469, -0.7656, 1.4844, 5.3320)
	# create_data()
	# print('---- started 03 ----')
	# with open("/Users/chloe/Documents/RushHour/scripts/node_list_03.pickle", "r") as fp:
	# 	node_list = pickle.load(fp)
	# print(type(node_list[0]))
	# print(node_list[0])
	# with open("/Users/chloe/Documents/RushHour/scripts/node_list.pickle", "r") as fp:
	# 	expected_list = pickle.load(fp)
	# print(type(expected_list[0]))
	# print(expected_list[0])






