from graphviz import Digraph
from json import dump,load
import Graph
from networkx import DiGraph, simple_cycles

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
	for i in range(red.end[0], board.width): # obstacles in front of red, include red
		cur_car = board.board_dict[(i, red.end[1])]
		if cur_car is not None: #exists on board
			# print("queue append: " + cur_car.tag)
			queue.append(cur_car)
			if cur_car.tag != 'r':
				red.edge_to.append(cur_car)
				cur_car.visited = True
	if len(queue) != 1: # continue searching only if red front is not empty
		i = red.start[0] - 1 # obstacles behind red
		while i >= 0:
			cur_car = board.board_dict[(i, red.start[1])]
			if cur_car is not None: #exists on board, not include red
				if cur_car.tag != 'r' and cur_car.visited != True: # can be horizontal
					queue.append(cur_car)
					# print("queue append: " + cur_car.tag)
					red.edge_to.append(cur_car)
					cur_car.visited = True
			i -= 1
	red.visited = True
	finished_list.append(red)
	while len(queue) != 0: # obstacle blockers
		cur_car	= queue.pop()
		# print('pop ', cur_car.tag)
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
		finished_list.append(cur_car) # list to be returned, all cars in MAG
	# clean all visited flags
	for i in range(0, board.height):
		for j in range(0, board.width):
			cur_car = board.board_dict[(i, j)]
			if cur_car is not None:
				cur_car.visited = False
	return finished_list
	

def visualize_mag(finished_list, filename):	
	dot = Digraph(format='png')
	# dot.node('dummy')
	dot.node('r',label='R')
	# dot.edge('dummy', 'r')
	for f in finished_list:
		dot.node(f.tag)
		#print(f.tag)
		for edge_car in f.edge_to:
			dot.node(edge_car.tag)
			dot.edge(f.tag, edge_car.tag)
			#print("edge: " + edge_car.tag)
	# save and show 
	dot.render(filename, view=False)


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

def list_to_graph(finished_list): #convert list to graph
	g = Graph.Graph(9)
	#print(len(finished_list))
	for car in finished_list:
		if car.tag == 'r':
			car_tag = len(finished_list) - 1
		else: 
			car_tag = int(car.tag)
		for adj_car in car.edge_to:
			if adj_car.tag == 'r':
				adj_car_tag = len(finished_list) - 1
			else:
				adj_car_tag = int(adj_car.tag)
			g.addEdge(car_tag, adj_car_tag)
	return g

def find_SCC(finished_list): 
# print and count number of SCC, return allscc list, return max scc len
	g = list_to_graph(finished_list)
	return g.find_SCC()

def find_cycles(finished_list): 
# print all cycles
	g = list_to_graph(finished_list)
	return g.all_cycles()

def num_cycles(finished_list): # number of cycles
	g = list_to_graph(finished_list)
	return len(g.all_cycles())

def num_in_cycles(finished_list): # number of small cycles in larger cycles
	g = list_to_graph(finished_list)
	return g.cycle_in_cycle()

def longest_path(finished_list): # longest path from red, len and paths
	g = list_to_graph(finished_list)
	return g.longest_path(len(finished_list) - 1)

def replace(elist, old_e, new_e): # replace a certain element in list
	for i in range(len(elist)):
		row = elist[i]
		for j in range(len(row)):
			e = row[j]
			if e == old_e:
				elist[i][j] = new_e
	return elist

def replace_1d(elist, old_e, new_e): #replace function for 1d list
	for i in range(len(elist)):
		e = elist[i]
		if e == old_e:
			elist[i] = new_e
	return elist

def num_nodes_in_cycle(finished_list): # return number of nodes in cycles
	g = list_to_graph(finished_list)
	_, cycle_list, _ = g.all_cycles() # list of all cycles
	cycle_nodes = [] # list of nodes belong to some cycle
	# iterate through cycle list to collect nodes
	for i in range(len(cycle_list)):
		row = cycle_list[i]
		for j in range(len(row)):
			e = row[j]
			if e not in cycle_nodes:
				cycle_nodes.append(e)
	# return number of nodes in cycles, and list of nodes in cycle
	return len(cycle_nodes), cycle_nodes

def pro_nodes_in_cycle(finished_list): # proportion of nodes in cycles
	num, _ = num_nodes_in_cycle(finished_list) # get number of nodes in cycles
	n_nodes, _ = get_mag_attr(finished_list) # total number of nodes
	return format(float(num)/n_nodes, '.2f')

def e_by_n(finished_list): # #edges / #nodes
	n_nodes, n_edges = get_mag_attr(finished_list)
	return format(float(n_edges)/n_nodes, '.2f')

def e_by_pn(finished_list): # #edges / (#nodes - #leaf_nodes)
	n_nodes, n_edges = get_mag_attr(finished_list)
	leaf_list = [] # list of leaf nodes
	# iterate through finished_list to find all leaf nodes
	for i in finished_list:
		if len(i.edge_to) == 0 and i not in leaf_list:
			leaf_list.append(i)
	return format(float(n_edges)/(n_nodes - len(leaf_list)), '.2f')

def e2_by_n(finished_list): # #edge^2 / #nodes
	n_nodes, n_edges = get_mag_attr(finished_list)
	return format(float(n_edges) * float(n_edges)/n_nodes, '.2f')

def global_cluster_coef(finished_list):
	g = list_to_graph(finished_list)
	return g.global_cluster_coef()

def av_local_cluster_coef(finished_list):
	g = list_to_graph(finished_list)
	return g.av_local_cluster_coef()






# testing

# my_car_list, my_red = json_to_car_list("/Users/chloe/Documents/RushHour/exp_data/data_adopted/prb21272.json")
# my_board = construct_board(my_car_list)
# new_car_list = construct_mag(my_board, my_red)
# visualize_mag(new_car_list, "/Users/chloe/Desktop/test_mag")

# n_node, n_edge = get_mag_attr(new_car_list)
# print("num_node: " + str(n_node))
# print("num_edge: " + str(n_edge))

# countscc, scclist, maxlen = find_SCC(new_car_list)
# scclist = replace(scclist, 8, 'R')
# print("num_scc:\n" + str(countscc) \
# 	+ "\nscc list:\n" + str(scclist) \
# 	+ "\nmax scc len:\n" +  str(maxlen))

# countc, clist, maxc = find_cycles(new_car_list)
# clist = replace(clist, 8, 'R')
# print("num_cycles:\n"+ str(countc)\
# 		+ "\ncycle list:\n" + str(clist)\
# 		+ "\nmax cycle len:\n"+ str(maxc))
# print("number of cycles in cycle: " \
# 		+ str(num_in_cycles(new_car_list)))

# depth, paths = longest_path(new_car_list)
# paths = replace(paths, 8, 'R')
# print("longest path len from red:" + str(depth) \
# 		+ "\npath:\n" + str(paths))

# n_nc, cycle_nodes = num_nodes_in_cycle(new_car_list)
# cycle_nodes = replace_1d(cycle_nodes, 8, 'R')
# pro = pro_nodes_in_cycle(new_car_list)
# print('number of nodes in cycles: ' + str(n_nc) \
# 		+ '\nlist of nodes in cycles: ' + str(cycle_nodes)\
# 		+ '\nproportion of nodes in cycles: ' + str(pro))

# ebn = e_by_n(new_car_list)
# ebpn = e_by_pn(new_car_list)
# e2bn = e2_by_n(new_car_list)
# print('#edges/#nodes = ' + str(ebn)\
# 		+ '\n#edges2/#nodes = ' + str(e2bn)
# 		+ '\n#edges/(#nodes - #leaf) = ' + str(ebpn))

# gcluster = global_cluster_coef(new_car_list)
# lcluster = av_local_cluster_coef(new_car_list)
# print('global cluster coef = ' + str(gcluster)\
# 		+ '\nmean local cluster coef = ' + str(lcluster))




