from BFS import *

def test_all_legal_moves(car_list, answer):
	all_moves = all_legal_moves(car_list, Board(car_list))
	assert len(all_moves) == answer, 'test_all_legal_moves FAILED, observe '+str(len(all_moves))

def test_InitializeChildren(node, params, answer):
	InitializeChildren(node, params)
	assert len(node.children) == answer, 'test_InitializedChildren FAILED, observe '+str(len(node.children))
	test_Node(node.children[1], params)

def test_move(car_list, car_tag, to_position, params, answer):
	print('-------------- test move ----------------')
	new_list, new_red = move(car_list, car_tag, to_position)
	new_node = Node(new_list, params)
	test_Node(new_node, params)
	test_InitializeChildren(new_node, params, answer)
	test_all_legal_moves(new_list, answer)

def test_red(node):
	assert node.red == node.board.red, 'test_red FAILED'

def test_Node(node, params):
	print('--------------- test node ---------------')
	print(node.board_to_str())
	test_red(node)
	print(node.value)
	for car in node.car_list:
		print('Car '+ car.tag 
			+ ', edge_to:'+str([str(i.tag) for i in car.edge_to])
			+ ', levels:'+str(car.level))

def test_is_solved(board, red, answer):
	assert is_solved(board, red) == answer, "test_is_solved FAILED"


if __name__ == '__main__':
	params = Params(0.7,0.6,0.5,0.4,0.3,0.2,0.1, 
					mu=0.0, sigma=1.0,
					feature_dropping_rate=0.0, 
					stopping_probability=0.05,
					pruning_threshold=10.0, 
					lapse_rate=0.05)
	instance = 'prb3217'
	ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
	car_list = json_to_car_list(ins_file)

	node = Node(car_list, params)

	MakeMove(node, params)
	sys.exit()
	
	# test_Node(node, params)
	# print()
	# for car in car_list:
	# 	print('Car '+ car.tag 
	# 		+ ', edge_to:'+str([str(i.tag) for i in car.edge_to])
	# 		+ ', levels:'+str(car.level))

	# test_all_legal_moves(car_list, 9)
	# test_is_solved(node.board, node.red, False)
	# test_InitializeChildren(node, params, 9)
	# test_move(car_list, '7', 6, params, 7)
	# new_list, new_red = move(car_list, '2', 0)
	# node = Node(new_list, params)
	# print(node.board_to_str())
	# test_is_solved(node.board, node.red, True)
	# test_move(car_list, '2', 0, params, 10)



	# for c in n.children:
	# 	print(c.board_to_str())
	# 	print('value='+str(c.value))
	# selected = MakeMove(n, params)
	# print('\n\nSelected Move:\n'+selected.board_to_str()+'value='+str(selected.value)+'\n')
	# for c in selected.children:
	# 	print(c.board_to_str()+'value='+str(c.value)+'\n')

	# sys.exit()
	# new_list, new_red = move_xy(new_list, 'r', 1, 2)
	# print(Node(new_list, params).board_to_str())
	# print(new_list)
	# new_list, new_red = move(new_list, '0', 18)
	# print(Node(new_list, params).board_to_str())
	# print(new_list)


	# trial_start = 2 # starting row number in the raw data
	# trial_end = 15 # inclusive
	# sub_data = recfromcsv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	# cur_carlist = None
	# cur_node = None
	# for i in range(trial_start-2, trial_end-1): 
	# 	row = sub_data[i]
	# 	if row['event'] == 'start':
	# 		instance = row['instance']
	# 		ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
	# 		initial_car_list = json_to_car_list(ins_file)
	# 		cur_carlist = initial_car_list
	# 		cur_node = Node(cur_carlist, params)
	# 		print('Instance '+str(instance))
	# 		print('Initial board \n'+cur_node.board_to_str())
	# 		print('----------------------------------')
	# 		continue
	# 	# make human move
	# 	piece = row['piece']
	# 	move_to = int(row['move'])
	# 	cur_carlist, _ = move(cur_carlist, piece, move_to) 
	# 	cur_node = Node(cur_carlist, params)
	# 	print('Current Board:\n'+cur_node.board_to_str())

	# cur_carlist, _ = move(cur_carlist, '3', 3)
	# cur_node = Node(cur_carlist, params)
	# print('Current Board:\n'+cur_node.board_to_str())
	
	# cur_node = MakeMove(cur_node, params)
	# print('Decision:\n'+cur_node.board_to_str())





# class Node:
# 	def __init__(self, cl):
# 		self.__car_list = cl # list of Car
# 		self.__children = []
# 		self.__value = None # float
# 		self.__red = None # Car
# 		self.__board = None # str
# 	def __members(self):
# 		return (self.__car_list, self.__children, self.__value, self.__red, self.__board)
# 	def __eq__(self, other):
# 		if type(other) is type(self):
# 			return self.__members() == other.__members()
# 		else:
# 			return False
# 	def __hash__(self):
# 		return hash(self.__members())
# 	def add_child(self, n):
# 		n.set_parent(self)
# 		self.__children.append(n)
# 	def set_parent(self, p):
# 		self.__parent = p
# 	def set_value(self, v):
# 		self.__value = v
# 	def get_carlist(self):
# 		return self.__car_list
# 	def get_red(self):
# 		if self.__red == None:
# 			for car in self.__car_list:
# 				if car.tag == 'r':
# 					self.__red = car
# 		return self.__red
# 	def get_board(self):
# 		return Board(self.__car_list)
# 	def get_value(self):
# 		if self.__value == None:
# 			self.__value = self.heuristic_value_function()
# 		return self.__value
# 	def get_child(self, ind):
# 		return self.__children[ind]
# 	def get_children(self):
# 		return self.__children
# 	def find_child(self, c):
# 		for i in range(len(self.__children)):
# 			if self.__children[i] == c:
# 				return i
# 		return None
# 	def find_child_by_str(self, bstr):
# 		for i in range(len(self.__children)):
# 			if self.__children[i].board_to_str() == bstr:
# 				return i
# 		return None
# 	def get_parent(self):
# 		return self.__parent
# 	def remove_child(self, c):
# 		for i in range(len(self.__children)):
# 			if self.__children[i] == c:
# 				c.parent = None
# 				self.__children.pop(i)
# 				return	
# 	def board_to_str(self):
# 		if self.__board == None:
# 			tmp_board = Board(self.__car_list)
# 			out_str = ''
# 			for i in range(tmp_board.height):
# 				for j in range(tmp_board.width):
# 					cur_car = tmp_board.board_dict[(j, i)]
# 					if cur_car == None:
# 						out_str += '.'
# 						if i == 2 and j == 5:
# 							out_str += '>'
# 						continue
# 					if cur_car.tag == 'r':
# 						out_str += 'R'
# 					else:
# 						out_str += cur_car.tag
# 					if i == 2 and j == 5:
# 						out_str += '>'
# 				out_str += '\n'
# 			self.__board = out_str
# 		return self.__board
# 	def heuristic_value_function(self):
# 		'''
# 		value = w0 * num_cars{MAG-level RED}
# 			+ w1 * num_cars{MAG-level 1} 
# 			+ w2 * num_cars{MAG-level 2}  
# 			+ w3 * num_cars{MAG-level 3} 
# 			+ w4 * num_cars{MAG-level 4} 
# 			+ w5 * num_cars{MAG-level 5} 
# 			+ w6 * num_cars{MAG-level 6}
# 			+ w7 * num_cars{MAG-level 7}  
# 			+ noise
# 		'''
# 		noise = np.random.normal(loc=params.mu, scale=params.sigma)
# 		# initialize MAG
# 		my_board2 = Board(self.__car_list)
# 		new_car_list2 = construct_mag(my_board2, my_board2.red)
# 		# each following level
# 		new_car_list2 = assign_level(new_car_list2)
# 		value = np.sum(np.array(get_num_cars_from_levels(new_car_list2, params.num_weights-1)) * np.array(params.weights))
# 		return value+noise
