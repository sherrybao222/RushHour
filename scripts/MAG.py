from graphviz import Digraph
from json import dump,load

class Car:
	start, end, length = [0,0], [0,0], [0,0] #hor,ver
	tag = ''
	puzzle_tag = ''
	orientation = ''
	edge_to = []
	visited = False
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
				self.board_dict[(i, j)] = None

def json_to_car_list(filename):
	with open(filename,'r') as data_file:
		car_list = []
		data = load(data_file)
		red = ''
		for c in data['cars']:
			cur_car = Car(s = [int(c['position'])%6, int(c['position']/6)], \
				l = int(c['length']), t = c['id'], o = c['orientation'], p = data['id'])
			car_list.append(cur_car)
			#print("car list : " + cur_car.tag + " start: " + str(cur_car.start) + " end: "  + str(cur_car.end) + " orientation: " + str(cur_car.orientation))
			if cur_car.tag == 'r':
				red = cur_car
	return car_list,red

def construct_board(car_list):
	board = Board()
	for car in car_list:
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
	return board

def construct_mag(board, red):
	#print(board.board_dict)
	queue = []
	finished_list = []
	i = board.width - 1
	for i in range(red.end[0], board.width): # obstacles in front of red
		cur_car = board.board_dict[(i, red.end[1])]
		if cur_car is not None: #exists on board
			#print("queue append: " + cur_car.tag)
			queue.append(cur_car)
			if cur_car.tag != 'r':
				red.edge_to.append(cur_car)
				cur_car.visited = True
	i = red.start[0] - 1 # obstacles behind red
	while i >= 0:
		cur_car = board.board_dict[(i, red.start[1])]
		if cur_car is not None: #exists on board
			#print("queue append: " + cur_car.tag)
			queue.append(cur_car)
			if cur_car.tag != 'r':
				red.edge_to.append(cur_car)
				cur_car.visited = True
		i -= 1
	red.visited = True
	finished_list.append(red)
	while len(queue) != 0: # obstacle blockers
		cur_car	= queue.pop()
		if cur_car.tag == 'r':
			continue
		#print('queue pop: ' + cur_car.tag)
		#for e in range(0, len(cur_car.edge_to)):
			#print("list contain: " + cur_car.edge_to[e].tag)
		if cur_car.orientation == 'vertical': # vertical
			if cur_car.start[1] > 0:
				j = cur_car.start[1] - 1
				while j >= 0: # upper
					meet_car =  board.board_dict[(cur_car.start[0], j)]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							#print("meet car: " + meet_car.tag)
							cur_car.edge_to.append(meet_car)
							#print('add edge to: ' + meet_car.tag)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
								#print("queue append " + meet_car.tag)
					j -= 1
			if cur_car.end[1] < board.height - 1:
				k = cur_car.end[1] + 1
				while k <= board.height - 1: # lower
					meet_car =  board.board_dict[(cur_car.start[0], k)]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							#print("meet car: " + meet_car.tag)
							cur_car.edge_to.append(meet_car)
							#print('add edge to: ' + meet_car.tag)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
								#print("queue append " + meet_car.tag)
					k += 1
		elif cur_car.orientation == 'horizontal': # or horizontal
			if cur_car.start[0] > 0:
				j = cur_car.start[0] - 1
				while j >= 0: # left
					#print("j = " + str(j))
					meet_car =  board.board_dict[(j, cur_car.start[1])]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							#print("meet car: " + meet_car.tag)
							cur_car.edge_to.append(meet_car)
							#print('add edge to: ' + meet_car.tag)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
								#print("queue append " + meet_car.tag)
					j = j - 1
			if cur_car.end[0] < board.width - 1:
				k = cur_car.end[0] + 1
				while k <= board.width - 1: # right
					meet_car =  board.board_dict[(k, cur_car.start[1])]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							#print("meet car: " + meet_car.tag)
							cur_car.edge_to.append(meet_car)
							#print('add edge to: ' + meet_car.tag)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
								#print("queue append " + meet_car.tag)
					k += 1
		cur_car.visited = True # mark
		finished_list.append(cur_car) # list to be returned
	# clean all visited flags
	for i in range(0, board.height):
		for j in range(0, board.width):
			cur_car = board.board_dict[(i, j)]
			if cur_car is not None:
				cur_car.visited = False
	return finished_list
	

def visualize_mag(finished_list, filename):	
	dot = Digraph()
	dot.node('dummy')
	dot.node('r')
	dot.edge('dummy', 'r')
	for f in finished_list:
		dot.node(f.tag)
		#print(f.tag)
		for edge_car in f.edge_to:
			dot.node(edge_car.tag)
			dot.edge(f.tag, edge_car.tag)
			#print("edge: " + edge_car.tag)
	# save and show 
	dot.render(filename, view=True)


def get_mag_attr(finished_list): # get num_node, num_edge
	num_node = 0
	num_edge = 0
	visited_node = []
	for f in finished_list:
		if f not in visited_node:
			visited_node.append(f)
			num_node += 1
		for edge_car in f.edge_to:
			num_edge += 1
			if edge_car not in visited_node:
				visited_node.append(edge_car)
				num_node += 1
	return num_node, num_edge

# testing

# my_car_list, my_red = json_to_car_list("/Users/chloe/Documents/RushHour/data/data_adopted/prb54506.json")
# my_board = construct_board(my_car_list)
# new_car_list = construct_mag(my_board, my_red)
# visualize_mag(new_car_list, "/Users/chloe/Desktop/test_mag.gv")
# n_node, n_edge = get_mag_attr(new_car_list)
# print("num_node: " + str(n_node))
# print("num_edge: " + str(n_edge))



