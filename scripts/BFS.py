# BFS model

import MAG, rushhour
import random, sys, copy, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph

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
		self.__value_bylevel = None
		self.__tag = None # parent car to be moved
		self.__pos1from = None # parent position
		self.__pos2from = None # parent position 
		self.__pos1to = None # self position as a child
		self.__pos2to = None # self position as a child

	def add_child(self, n):
		n.set_parent(self)
		self.__children.append(n)

	def move_from_parent(self, tag, pos1from, pos2from, pos1to, pos2to):
		''' if current node is a child, record how it is transformed from its parent 
		'''
		self.__tag = tag # car moved from parent
		self.__pos1from = pos1from # parent car position
		self.__pos2from	= pos2from # parent car position
		self.__pos1to = pos1to # current/child position
		self.__pos2to = pos2to # current/child position

	def get_move_from_parent(self):
		return self.__tag, self.__pos1from, self.__pos2from, self.__pos1to, self.__pos2to
		
	def get_move(self):
		return self.__tag, self.__pos1, self.__pos2, self.__pos1old, self.__pos2old

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
			# self.__value, self.__value_bylevel = Value1(self.__car_list, self.get_red())
			self.__value = Value1(self.__car_list, self.get_red())
		return self.__value

	def get_value_bylevel(self):
		return self.__value_bylevel

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
		# self.__value, self.__value_bylevel = Value1(self.__car_list, self.__red)		
		self.__value = Value1(self.__car_list, self.__red)		

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
	# value_level = []
	noise = np.random.normal(loc=mu, scale=sigma)
	value = 0
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# number of cars at the top level (red only)
	value += w0 * 1 
	# value_level.append(1)
	# each following level
	for j in range(num_parameters - 1): 
		level = j+1
		new_car_list2 = MAG.clean_level(new_car_list2)
		new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
		cars_from_level = MAG.get_cars_from_level2(new_car_list2, level)
		value += weights[level] * (len(cars_from_level))
		# value_level.append(len(cars_from_level))
	return value+noise



def Manhattan_Value(car_list2, red):
	''' value function is manhattan distance,
		start by converting car_list to instance,
		need to import Zahy's rushhour.py functions
	'''
	h,v = {},{}
	name = ''
	for car in car_list2:
		if car.orientation == 'horizontal':
			h[car.tag] = (car.start[0], car.start[1], car.length)
		elif car.orientation == 'vertical':
			v[car.tag] = (car.start[0], car.start[1], car.length)
	cur_ins = rushhour.RHInstance(h,v,name)
	value = rushhour.min_manhattan_distance_calc(cur_ins)
	# print('following board: manhattan distance '+str(value))
	# print('board:\n'+Node(car_list2).board_to_str())
	return -value



def DropFeatures(delta):
	pass



def Lapse(probability=0.1): 
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
		tmp = copy.deepcopy(root_node)
		all_moves = MAG.all_legal_moves(tmp.get_carlist(), tmp.get_red(), tmp.get_board())
		for i, (tag, pos) in enumerate(all_moves):
			new_list, _ = MAG.move2(tmp.get_carlist(), tag, pos[0], pos[1])
			for car in tmp.get_carlist():
				if car.tag == tag:
					pos1from, pos2from = car.start[0], car.start[1]
			dummy_child = Node(new_list)
			dummy_child.move_from_parent(tag, pos1from, pos2from, pos[0], pos[1])
			root_node.add_child(dummy_child)



def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	traversed = []
	while len(n.get_children()) != 0:
		n = ArgmaxChild(n)
		traversed.append(n)
	return n, traversed
 


def ExpandNode(node, threshold):
	''' create all possible nodes under input node, 
	cut the ones below threshold '''
	if len(node.get_children()) == 0:
		InitializeChildren(node)
	Vmax = MaxChildValue(node)
	for child in node.get_children():
		if abs(child.get_value() - Vmax) > threshold:
			node.remove_child(child)
	return ArgmaxChild(node)



def Backpropagate(this_node, root_node):
	''' update value back until root node '''
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
	return maxChild



def MakeMove(state, delta=0, gamma=0.05, theta=float('inf')):
	''' returns an optimal move to make next, given current state '''
	root = state # state is already a node
	InitializeChildren(root)
	if Lapse():
		# print('Random move made')
		return RandomMove(root), [], []
	else:
		DropFeatures(delta)
		considered_node = [] # nodes already traversed along the branch in this ieration
		considered_node2 = [] # new node expanded along the branch in this iteration
		while not Stop(probability=gamma) and not Determined(root):
			n, traversed = SelectNode(root)
			considered_node.append(traversed)
			n2 = ExpandNode(n, theta)
			considered_node2.append(n2)
			if not plot_tree_flag: 
				# if plot decision tree, do not show backproped value
				# so that a comparison of values among children can be observed directly
				Backpropagate(n, root)
			if n2.get_value() == 0: 
				# terminate the algorithm if found a terminal node
				break
	print('move made')
	return ArgmaxChild(root), considered_node, considered_node2



def estimate_prob(root_node, expected_board='', iteration=100):
	''' Estimate the probability of next possible moves given the root node '''
	InitializeChildren(root_node)
	frequency = [0] * len(root_node.get_children())
	for i in range(iteration):
		new_node, _, _ = MakeMove(root_node)
		child_idx = root_node.find_child(new_node)
		frequency[child_idx] += 1
	
	frequency = np.array(frequency, dtype=np.float32)/iteration # now frequency becomes probability
	chance = float(1)/float(len(frequency))

	sol_idx = None
	sol_value_level = None
	# print('initial board:\n'+str(root_node.board_to_str()))
	for i in range(len(root_node.get_children())):
		if root_node.get_child(i).board_to_str() == expected_board:
			sol_idx = i
			sol_value_level = root_node.get_child(i).get_value_bylevel()
		# print('possible move:\n'+str(root_node.get_child(i).board_to_str()))
		# print('probability:'+str(frequency[i])+'\n')

	return root_node.get_children(), frequency, sol_idx, sol_value_level, chance


def ibs(root_node, expected_board=''):
	''' inverse binomial sampling 
		return the number of simulations until hit target
	'''
	InitializeChildren(root_node)
	frequency = [0] * len(root_node.get_children())
	num_simulated = 0
	hit = False
	sol_idx = None
	
	while not hit:
		new_node, _, _ = MakeMove(root_node)
		child_idx = root_node.find_child(new_node)
		frequency[child_idx] += 1
		num_simulated += 1
		if root_node.get_child(child_idx).board_to_str() == expected_board:
			sol_idx = child_idx
			hit = True
	
	frequency = np.array(frequency, dtype=np.float32)
	# chance = float(1)/float(len(frequency))

	return num_simulated, root_node.get_children(), frequency, sol_idx


	
	

def plot_trial(this_w0=0, this_w1=0, this_w2=0, this_w3=-1, this_w4=0, this_w5=0, this_w6=0, this_w7=0, this_mu=0, this_sigma=1):
	''' show movie of the model's consideration of next moves (with tree expansion)
		based on subject's current position from a trial'''
	global w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma, weights
	w0 = this_w0
	w1 = this_w1
	w2 = this_w2
	w3 = this_w3
	w4 = this_w4
	w5 = this_w5
	w6 = this_w6
	w7 = this_w7
	mu = this_mu
	sigma = this_sigma
	weights = [w0, w1, w2, w3, w4, w5, w6, w7]
	trial_start = 2 # starting row number in the raw data
	trial_end = 20 # ending row number in the raw data
	global plot_tree_flag # whether to visialize the tree at the same time
	plot_tree_flag = True
	sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	os.chdir(dir_name)

	# construct initial node
	first_line = sub_data.loc[trial_start-2,:]
	instance = first_line['instance']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	initial_node = Node(initial_car_list)
	# print('Initial board:\n'+initial_node.board_to_str())
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1 # move number in this human trial
	img_count = 0 # image count


	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		
		# plot text
		plot_blank(instance, img_count, text='Move Number '+str(move_num), color='orange')
		# initialize tree plot if required
		if plot_tree_flag:
			dot = Digraph(comment='Test Tree', format='jpg', strict=True)
			dot.attr(size='10,10', fixedsize='true')
			dot.node(str(id(cur_node)), str(cur_node.get_value()))
		img_count += 1
		# plot blank space
		plot_blank(instance, img_count, text='Initial Board', color='orange')
		img_count += 1
		# plot initial board
		plot_state(cur_node, instance, img_count)
		img_count += 1


		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']


		# run model to make one decision (which contains many iterations)
		selectedmove, considered, considered2 = MakeMove(cur_node)
		# print('Initial board:\n'+cur_node.board_to_str())
		# print('move made:\n'+selectedmove.board_to_str())
		total_iteration = len(considered2)
		cur_iteration_num = 1 # initialize iteration count


		# plot each itertaion seperately
		if considered == []:
			# plot text
			plot_blank(instance, img_count, text='Random Move', color='green')
			img_count += 1
		for pos, pos2 in zip(considered, considered2): # if any iteration is considered
			# plot text
			plot_blank(instance, img_count, text='Iteration '+str(cur_iteration_num)+'/'+str(total_iteration), color='blue')
			cur_iteration_num += 1
			img_count += 1
			# plot board
			plot_state(cur_node, instance, img_count) # initial state
			img_count += 1
			# plot the node traversed along the selected branch in this iteration
			tree_cur = cur_node
			for pos_cur in pos: 
				plot_state(pos_cur, instance, img_count)
				if plot_tree_flag:
					for child in tree_cur.get_children():
						dot.node(str(id(child)), str(child.get_value()))
						dot.edge(str(id(tree_cur)), str(id(child)))
					tree_cur = pos_cur
				img_count += 1
			# plot the new node expanded along this branch in this iteration
			plot_state(pos2, instance, img_count)
			if plot_tree_flag:
				for child in tree_cur.get_children():
					dot.node(str(id(child)), str(child.get_value()))
					dot.edge(str(id(tree_cur)), str(id(child)))
				tree_cur = pos2
				for child in tree_cur.get_children():
					dot.node(str(id(child)), str(child.get_value()))
					dot.edge(str(id(tree_cur)), str(id(child)))
			img_count += 1
			

		# plot selected move 
		plot_blank(instance, img_count, text='Selected Move', color='green')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		plot_state(selectedmove, instance, img_count)
		if plot_tree_flag:
			dot.edge(str(id(cur_node)), str(id(selectedmove)), color='red')
			dot.render('/Users/chloe/Desktop/RHfig/'+instance+'_'+str(img_count)+'_tree', 
						view=False)
		img_count += 1


		# plot actual move made by human 
		plot_blank(instance, img_count, text='Human Move', color='red')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1

		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_board, _ = MAG.construct_board(cur_carlist)
		cur_node = Node(cur_carlist)
		plot_state(cur_node, instance, img_count)
		img_count += 1


		# make movie and save
		move_num += 1
		make_movie(move_num-1, format='avi')
		
		
		# clean all jpg files after movie done
		test = os.listdir(dir_name)
		for item in test:
		    if item.endswith("board.jpg") or item.endswith('tree'):
		        os.remove(os.path.join(dir_name, item))
		

		# sys.exit()
		



def hist_prob(this_w0=0, this_w1=0, this_w2=0, this_w3=-1, this_w4=0, this_w5=0, this_w6=0, this_w7=0, this_mu=0, this_sigma=1):
	''' visualize histogram of the probability of human move'''
	global w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma, weights
	w0 = this_w0
	w1 = this_w1
	w2 = this_w2
	w3 = this_w3
	w4 = this_w4
	w5 = this_w5
	w6 = this_w6
	w7 = this_w7
	mu = this_mu
	sigma = this_sigma
	weights = [w0, w1, w2, w3, w4, w5, w6, w7]
	global plot_tree_flag # whether to visialize the tree at the same time
	plot_tree_flag = False
	trial_start = 2 # starting row number in the raw data
	trial_end = 20 # ending row number in the raw data
	sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	os.chdir(dir_name)

	# construct initial node
	first_line = sub_data.loc[trial_start-2,:]
	instance = first_line['instance']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	initial_node = Node(initial_car_list)
	print('Initial board:\n'+initial_node.board_to_str())
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1 # move number in this human trial


	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']
		
		# estimate probability
		_, frequency, sol_idx, _, _ = estimate_prob(cur_node, expected_board=Node(MAG.move(cur_carlist, piece, move_to)[0]).board_to_str())
		
		# plot histogram
		barlist = plt.bar(np.arange(len(frequency)), frequency)
		barlist[sol_idx].set_color('r')
		plt.ylim(top=1.0, bottom=0)
		plt.title('Instance '+instance+', move number '+str(move_num))
		plt.savefig(dir_name+instance+'_hist_move_'+str(move_num)+'.jpg')
		plt.close()

		# make human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_board, _ = MAG.construct_board(cur_carlist)
		cur_node = Node(cur_carlist)

		move_num += 1

		




def plot_trial_human(trial_start=21, trial_end=53):
	''' show movie of a human trial
		trial_start: starting row number in the raw data
		trial_end: ending row number in the raw data
	'''
	sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	os.chdir(dir_name)
	global plot_tree_flag # whether to visialize the tree at the same time
	plot_tree_flag = False

	# construct initial node
	first_line = sub_data.loc[trial_start-2,:]
	instance = first_line['instance']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	initial_node = Node(initial_car_list)
	print('Initial board:\n'+initial_node.board_to_str())
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	global move_num # move number in this human trial
	move_num = 1
	img_count = 0 # image count

	# plot blank space
	plot_blank(instance, img_count, text='Human Trial, '+instance, color='orange', imgtype='human')
	img_count += 1
	# plot initial board
	plot_state(cur_node, instance, img_count, imgtype='human')
	img_count += 1

	# every human move in the trial
	for i in range(trial_start-1, trial_end-1):
	
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']

		# make move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_board, _ = MAG.construct_board(cur_carlist)
		cur_node = Node(cur_carlist)
		move_num += 1
		
		# plot current human move 
		plot_state(cur_node, instance, img_count, imgtype='human')
		img_count += 1

	# make movie and save
	make_movie(1, format='avi', imgtype='human')
	
	
	# clean all jpg files after movie done
	test = os.listdir(dir_name)
	for item in test:
	    if item.endswith("human.jpg"):
	        os.remove(os.path.join(dir_name, item))
	






def heat_map(this_w0=0, this_w1=0, this_w2=0, this_w3=-1, this_w4=0, this_w5=0, this_w6=0, this_w7=0, this_mu=0, this_sigma=1):
	''' show the model's consideration of immediate next moves 
		by heatmap and arrows,
		based on subject's current position from a trial '''
	global w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma, weights
	w0 = this_w0
	w1 = this_w1
	w2 = this_w2
	w3 = this_w3
	w4 = this_w4
	w5 = this_w5
	w6 = this_w6
	w7 = this_w7
	mu = this_mu
	sigma = this_sigma
	weights = [w0, w1, w2, w3, w4, w5, w6, w7]
	global plot_tree_flag # whether to visialize the tree at the same time
	plot_tree_flag = False
	trial_start = 2 # starting row number in the raw data
	trial_end = 20 # ending row number in the raw data
	sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	os.chdir(dir_name)

	# construct initial node
	first_line = sub_data.loc[trial_start-2,:]
	instance = first_line['instance']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	initial_node = Node(initial_car_list)
	print('Initial board:\n'+initial_node.board_to_str())
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	global move_num # move number in this human trial
	move_num = 1
	img_count = 0


	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']
		
		# estimate probability
		children_list, frequency, sol_idx, _, _ = estimate_prob(cur_node, expected_board=Node(MAG.move(cur_carlist, piece, move_to)[0]).board_to_str(), iteration=50)
		
		# plot heatmap
		plot_heatmap(cur_node, instance, img_count, children_list, frequency, sol_idx, imgtype='heatmap')
		img_count += 1

		# make human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_board, _ = MAG.construct_board(cur_carlist)
		cur_node = Node(cur_carlist)

		move_num += 1



def str_to_matrix(string):
	''' convert string of board to a int matrix '''
	matrix = np.zeros((6,6), dtype=int)
	line_idx = 0
	for line in string.split('\n'):
		char_idx = 0
		for char in line:
			if char == '>':
				continue
			elif char == '.':
				matrix[line_idx][char_idx] = -1
			elif char == 'R':
				matrix[line_idx][char_idx] = 0
			else:
				matrix[line_idx][char_idx] = int(char)+1
			char_idx += 1
		line_idx += 1
	return matrix



def plot_state(cur_node, instance, idx, out_file='/Users/chloe/Desktop/RHfig/', imgtype='board'):
	''' visualize the current node/board configuration '''
	matrix = str_to_matrix(cur_node.board_to_str())
	matrix = np.ma.masked_where(matrix==-1, matrix)
	cmap = plt.cm.Set1
	cmap.set_bad(color='white')
	fig, ax = plt.subplots()
	im = ax.imshow(matrix, cmap=cmap)
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			num = matrix[i, j]
			if num == 0:
				num = 'R'
			elif num > 0:
				num -= 1
			else:
				num = ''
			text = ax.text(j, i, num, ha="center", va="center", color="black", fontsize=14)
	if imgtype == 'human':
		plt.title('Move '+str(move_num-1))
	plt.savefig(out_file+instance+'_'+str(idx)+'_'+imgtype+'.jpg')
	plt.close()


import scipy.stats as stats
def plot_heatmap(cur_node, instance, idx, children_list, frequency, sol_idx, out_file='/Users/chloe/Desktop/RHfig/', imgtype='board'):
	''' visualize the arrows of heatmap  '''
	matrix = str_to_matrix(cur_node.board_to_str())
	matrix = np.ma.masked_where(matrix==-1, matrix)
	cmap = plt.cm.Set1
	cmap.set_bad(color='white')
	fig, ax = plt.subplots()
	im = ax.imshow(matrix, cmap=cmap)
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			num = matrix[i, j]
			if num == 0:
				num = 'R'
			elif num > 0:
				num -= 1
			else:
				num = ''
			text = ax.text(j, i, num, ha="center", va="center", color="black", fontsize=14)
	count = 0
	for child in children_list:
		tag, pos1from, pos2from, pos1to, pos2to = child.get_move_from_parent()
		if count == sol_idx:
			plt.arrow(x=pos1from, y=pos2from, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
					head_width=0.15, head_length=0.1, alpha=0.8, color='red',
					lw= 20 * frequency[count])
		else:
			plt.arrow(x=pos1from, y=pos2from, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
					head_width=0.15, head_length=0.1, alpha=0.5, color='black',
					lw= 20 * frequency[count])
		count += 1
	if imgtype == 'heatmap':
		plt.title('Heatmap Move '+str(move_num-1))
	plt.savefig(out_file+instance+'_'+str(idx)+'_'+imgtype+'.jpg')
	plt.close()



def plot_blank(instance, idx, text, color, out_file='/Users/chloe/Desktop/RHfig/', imgtype='board'):
	''' plot a blank image
		with the entire page filled by one color and a text message '''
	fig, ax = plt.subplots()
	fig.patch.set_facecolor(color)
	fig.patch.set_alpha(0.1)
	ax.patch.set_facecolor(color)
	ax.patch.set_alpha(0.1)
	ax.text(0.3, 0.5, text, fontsize=20)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.axis('off')
	fig.savefig(out_file+instance+'_'+str(idx)+'_'+imgtype+'.jpg', \
				facecolor=fig.get_facecolor(), edgecolor='none')
	plt.close()



import imageio
from pprint import pprint
import time
import datetime
import cv2
import datetime
def make_movie(move_num, path='/Users/chloe/Desktop/RHfig/', format='gif', imgtype='board'):
	''' make a movie using png files '''
	os.chdir(path)
	if format == 'gif':
		e=sys.exit
		duration = 0.5
		filenames = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('_'+imgtype+'.jpg')]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
		images = []
		for filename in filenames:
			images.append(imageio.imread(filename))
			output_file = 'MOVIE-'+imgtype+'-%s.gif' % (str(move_num)+'-'+datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
		imageio.mimsave(output_file, images, duration=duration)
	elif format == 'avi':
		image_folder = path
		video_name = 'MOVIE-'+imgtype+'-%s.avi' % (str(move_num)+'-'+datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
		images = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('_'+imgtype+'.jpg')]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape
		video = cv2.VideoWriter(video_name, 0, 1, (width,height))
		for image in images:
		    video.write(cv2.imread(os.path.join(image_folder, image)))
		cv2.destroyAllWindows()
		video.release()


# import libraries
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
import pymc3 as pm3
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

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
weights_todo = [w1]
num_parameters = len(weights)
sim_num = 1
trial_start = 2 # starting row number in the raw data
trial_end = 20
global move_num # move number in this human trial
global plot_tree_flag # whether to visialize the tree at the same time
plot_tree_flag = True
global plot_flag # whether to plot
plot_flag = True

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



def my_ll(weights=weights):
	print('simulation')

	ll = 0

	# construct initial node
	first_line = sub_data.loc[trial_start-2,:]
	instance = first_line['instance']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	initial_node = Node(initial_car_list)
	# print('Initial board:\n'+initial_node.board_to_str())
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	global move_num # move number in this human trial
	move_num = 1


	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']
		
		children_list, frequency, sol_idx, _, _ = estimate_prob(cur_node, 
			expected_board=Node(MAG.move(cur_carlist, piece, move_to)[0]).board_to_str(), 
			iteration=5)
		ll += -np.log(frequency[sol_idx])
		# num_simulated, _, _, _ = ibs(cur_node, expected_board=Node(MAG.move(cur_carlist, piece, move_to)[0]).board_to_str())
		# print(num_simulated)
		# ll += -harmonic_sum(num_simulated)

		# make human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_board, _ = MAG.construct_board(cur_carlist)
		cur_node = Node(cur_carlist)

		move_num += 1


def harmonic_sum(n):
	''' return sum of harmonic series from 1 to k '''
	i = 1
	s = 0.0
	for i in range(1, n+1):
		s += 1/i
	return s


# results = minimize(my_ll, weights, 
# 		method='Nelder-Mead', options={'disp': True})	
# print(results)


plot_model()






