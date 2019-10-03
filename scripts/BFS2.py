# BFS model

import MAG, rushhour
import random, sys, copy, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph
from scipy.optimize import minimize
import cProfile, pstats, StringIO
from operator import methodcaller
import multiprocessing as mp


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
		tmp_b, _ = MAG.construct_board(self.__car_list)
		return tmp_b

	def get_value(self):
		if self.__value == None:
			self.__value = Value1(self.__car_list, self.get_red())
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
			self.__board = out_str
		return self.__board



def Value0(car_list2, red):
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
	noise = np.random.normal(loc=mu, scale=sigma)
	value = 0
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# number of cars at the top level (red only)
	value += w0 * 1 
	# each following level
	new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
	for j in range(num_parameters - 1): 
		level = j+1
		# new_car_list2 = MAG.clean_level(new_car_list2)
		cars_from_level = MAG.get_cars_from_level2(new_car_list2, level)
		value += weights[level] * (len(cars_from_level))
	return value+noise



def Value1(car_list2, red):
	'''
	value = w0 * num_cars{MAG-level 0/RED}
		+ w1 * num_cars{MAG-level 1} 
		+ w2 * num_cars{MAG-level 2}  
		+ w3 * num_cars{MAG-level 3} 
		+ w4 * num_cars{MAG-level 4} 
		+ w5 * num_cars{MAG-level 5} 
		+ w6 * num_cars{MAG-level 6}
		+ w7 * num_cars{MAG-level 7}  
		+ noise
	'''
	noise = np.random.normal(loc=mu, scale=sigma)
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# each following level
	new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
	value = np.sum(np.array(MAG.get_num_cars_from_levels(new_car_list2, num_parameters)) * np.array(weights))
	return value+noise


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
	return MAG.check_win(root_node.get_board(), root_node.get_red())



def RandomMove(node):
	''' make a random move and return the resulted node '''
	return random.choice(node.get_children())
	


def InitializeChildren(root_node):
	''' initialize the list of children (using all possible moves) '''
	if len(root_node.get_children()) == 0:
		all_moves = MAG.all_legal_moves(root_node.get_carlist(), root_node.get_red(), root_node.get_board())
		root_car_list = root_node.get_carlist()
		for i, (tag, pos) in enumerate(all_moves):
			new_list, _ = MAG.move2(root_car_list, tag, pos[0], pos[1])
			dummy_child = Node(new_list)
			root_node.add_child(dummy_child)



def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	# traversed = []
	while len(n.get_children()) != 0:
		n = ArgmaxChild(n)
		# traversed.append(n)
	# return n, traversed
	return n, []
 


def ExpandNode(node, threshold):
	''' create all possible nodes under input node, 
	cut the ones below threshold '''
	if len(node.get_children()) == 0:
		InitializeChildren(node)
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
	# maxChild = None
	# for child in root_node.get_children():
	# 	if maxChild == None:
	# 		maxChild = child
	# 	elif maxChild.get_value() < child.get_value():
	# 		maxChild = child
	# return maxChild
	return max(root_node.get_children(), key=methodcaller('get_value'))
	



def MakeMove(state, delta=0, gamma=0.05, theta=10):
	''' returns an optimal move to make next, given current state '''
	root = state # state is already a node
	InitializeChildren(root)
	if Lapse():
		# print('Random move made')
		return RandomMove(root), [], []
	else:
		DropFeatures(delta)
		# considered_node = [] # nodes already traversed along the branch in this ieration
		considered_node2 = [] # new node expanded along the branch in this iteration
		while not Stop(probability=gamma) and not Determined(root):
			# n, traversed = SelectNode(root)
			n, _ = SelectNode(root)
			# considered_node.append(traversed)
			n2 = ExpandNode(n, theta)
			considered_node2.append(n2)
			Backpropagate(n, root)
			if n2.get_value() >= abs(np.random.normal(loc=mu, scale=sigma)): # terminate the algorithm if found a terminal node
				break
	# return ArgmaxChild(root), considered_node, considered_node2
	return ArgmaxChild(root), [], considered_node2



def estimate_prob(root_node, expected_board='', iteration=100):
	''' Estimate the probability of next possible moves given the root node '''
	# InitializeChildren(root_node)
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
	
	frequency = np.array(frequency, dtype=np.float32)/iteration # now frequency becomes probability

	for i in range(len(root_node.get_children())):
		if root_node.get_child(i).board_to_str() == expected_board:
			sol_idx = i

	return root_node.get_children(), frequency, sol_idx, [], []



def ibs(root_node, expected_board=''):
	''' inverse binomial sampling 
		return the number of simulations until hit target
	'''
	InitializeChildren(root_node)
	# frequency = [0] * len(root_node.get_children())
	num_simulated = 0
	hit = False
	# sol_idx = None
	while not hit:
		new_node, _, _ = MakeMove(root_node)
		# child_idx = root_node.find_child(new_node)
		# frequency[child_idx] += 1
		num_simulated += 1
		if new_node.board_to_str() == expected_board:
			# sol_idx = child_idx
			hit = True
	# frequency = np.array(frequency, dtype=np.float32)
	# chance = float(1)/float(len(frequency))
	# return num_simulated, root_node.get_children(), frequency, sol_idx
	return num_simulated


def harmonic_sum(n):
	''' return sum of harmonic series from 1 to k '''
	i = 1
	s = 0.0
	for i in range(1, n+1):
		s += 1.0/i
	return s




global w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma, weights, num_parameters
w0 = 0
w1 = -5
w2 = -4
w3 = -3
w4 = -1
w5 = -1
w6 = -1
w7 = -1
mu = 0
sigma = 1
weights = [w0, w1, w2, w3, w4, w5, w6, w7]
num_parameters = len(weights)
sim_num = 1
trial_start = 21 # starting row number in the raw data
trial_end = 53
global move_num # move number in this human trial
global plot_tree_flag # whether to visialize the tree at the same time
plot_tree_flag = True

sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
# construct initial node
first_line = sub_data.loc[trial_start-2,:]
instance = first_line['instance']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
ins_file = ins_dir + instance + '.json'
initial_car_list, _ = MAG.json_to_car_list(ins_file)
initial_node = Node(initial_car_list)
print('========================== started =======================')

# initialize parameters
node_list = [] # list of node from data
expected_list = [] # list of expected human move node, str
cur_node = initial_node
cur_carlist = initial_car_list

# save data in advance
for i in range(trial_start-1, trial_end-2):
	# load data from datafile
	row = sub_data.loc[i, :]
	piece = row['piece']
	move_to = row['move']

	# create human move
	cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
	node_list.append(cur_node)

	# make human move node
	cur_node = Node(cur_carlist)
	expected_list.append(cur_node.board_to_str())




def my_ll(weights=weights): # parallel computing

	# pr = cProfile.Profile()
	# pr.enable()
	print('--------------------- simulation --------------------')

	all_ibs = [pool.apply(ibs, args=(cur, exp_str)) 
			for cur, exp_str in zip(copy.deepcopy(node_list), expected_list)]
	print('all_ibs ', all_ibs)
	all_ll = pool.map(harmonic_sum, [n for n in all_ibs])
	print('all_ll ', all_ll)


	# pr.disable()
	# s = StringIO.StringIO()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	# ps.print_stats()
	# print s.getvalue()
		
	return -np.sum(all_ll)



def my_ll_sequential(weights=weights): # non-parallel

	pr = cProfile.Profile()
	pr.enable()
	print('simulation')
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1
	ll = 0

	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		# load data from datafile
		# print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']
		
		# children_list, frequency, sol_idx, _, _ = estimate_prob(cur_node, 
		# 	expected_board=Node(MAG.move(cur_carlist, piece, move_to)[0]).board_to_str(), 
		# 	iteration=5)
		# ll += -np.log(frequency[sol_idx])

		# MakeMove(cur_node)
		# break

		# create human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)

		# log likelihood calculation
		num_simulated = ibs(cur_node, 
				expected_board=Node(cur_carlist).board_to_str())
		ll += harmonic_sum(num_simulated)
		# print('ll: '+str(ll))
		print('ibs: '+str(num_simulated))

		# make human move node
		cur_node = Node(cur_carlist)
		move_num += 1

	pr.disable()
	s = StringIO.StringIO()
	sortby = 'cumulative'
	ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	ps.print_stats()
	print s.getvalue()

	return -ll



def plot_trial():
	''' show movie of the model's consideration of next moves (with tree expansion)
		based on subject's current position from a trial'''
	os.chdir(dir_name)
	pr = cProfile.Profile()
	pr.enable()
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1 # move number in this human trial
	img_count = 1


	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		
		# initialize tree plot if required
		if plot_tree_flag:
			dot = Digraph(comment='Test Tree', format='jpg', strict=True)
			dot.attr(size='50,50', fixedsize='true')
			dot.node(str(id(cur_node)), str(cur_node.get_value()))
		
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']

		# run model to make one decision (which contains many iterations)
		selectedmove, considered, considered2 = MakeMove(cur_node)
		total_iteration = len(considered2)
		cur_iteration_num = 1 # initialize iteration count

		# plot each itertaion seperately
		for pos, pos2 in zip(considered, considered2): # if any iteration is considered
			# plot the node traversed along the selected branch in this iteration
			tree_cur = cur_node
			for pos_cur in pos: 
				if plot_tree_flag:
					for child in tree_cur.get_children():
						dot.node(str(id(child)), str(child.get_value()))
						dot.edge(str(id(tree_cur)), str(id(child)))
					tree_cur = pos_cur
			# plot the new node expanded along this branch in this iteration
			if plot_tree_flag:
				for child in tree_cur.get_children():
					dot.node(str(id(child)), str(child.get_value()))
					dot.edge(str(id(tree_cur)), str(id(child)))
				tree_cur = pos2
				for child in tree_cur.get_children():
					dot.node(str(id(child)), str(child.get_value()))
					dot.edge(str(id(tree_cur)), str(id(child)))

		# plot selected move 
		if plot_tree_flag:
			dot.edge(str(id(cur_node)), str(id(selectedmove)), color='red')
			dot.render('/Users/chloe/Desktop/RHfig/'+instance+'_'+str(img_count)+'_tree', 
						view=False)

		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_node = Node(cur_carlist)

		# make movie and save
		move_num += 1
		img_count += 1
		
		# clean all jpg files after movie done
		test = os.listdir(dir_name)
		for item in test:
		    if item.endswith("board.jpg") or item.endswith('tree'):
		        os.remove(os.path.join(dir_name, item))

		pr.disable()
		s = StringIO.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print s.getvalue()



# plot_trial()
# sys.exit()



pr = cProfile.Profile()
pr.enable()
	
pool = mp.Pool(mp.cpu_count())
print(mp.cpu_count())
results = minimize(my_ll, weights, 
		method='Nelder-Mead', options={'disp': True})	
print(results)
pool.close()

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()



