from json import dump, load
from Car import *

class Board:
	def __init__(self, car_list):
		# TODO: change argument car_list to board_dict
		self.height = 6
		self.width = 6
		self.board_dict = {}
		for i in range(0, self.height):
			for j in range(0, self.width):
				self.board_dict[(j, i)] = None
		for car in car_list:
			if car.tag == 'r':
				self.red = car
			occupied_space = []
			if car.orientation == 'horizontal':
				for i in range(car.length):
					occupied_space.append((car.start[0] + i, car.start[1]))
			elif car.orientation == 'vertical':
				for i in range(car.length):
					occupied_space.append((car.start[0], car.start[1] + i))
			for xy in occupied_space:
				self.board_dict[xy] = car

def json_to_car_list(filename):
	with open(filename,'r') as data_file:
		car_list = []
		data = load(data_file)
		for c in data['cars']:
			cur_car = Car(s = [int(c['position'])%6, int(c['position']/6)], \
				l = int(c['length']), t = str(c['id']), o = c['orientation'], p = data['id'])
			car_list.append(cur_car)
	return car_list

def carlist_to_board_str(carlist):
	board = Board(carlist)
	out_str = ''
	for i in range(board.height):
		for j in range(board.width):
			cur_car = board.board_dict[(j, i)]
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

def move(car_list, car_tag, to_position): 
	'''
		make a move and return the new car list, 
		single position label
	'''
	new_list2 = []
	for cur_car in car_list:
		if cur_car.tag == car_tag:
			new_car = Car(s = [int(to_position)%6, int(to_position/6)],\
				l = int(cur_car.length), t = car_tag, \
				o = cur_car.orientation, p = cur_car.puzzle_tag)
		else:
			new_car = Car(s = [int(cur_car.start[0]), int(cur_car.start[1])], \
							l = int(cur_car.length), t = cur_car.tag,\
							o = cur_car.orientation, p = cur_car.puzzle_tag)
		new_list2.append(new_car)
		if new_car.tag == 'r':
			red = new_list2[-1]
	return new_list2, red

def move_xy(car_list, car_tag, to_position1, to_position2): 
	'''
		make a move and return the new car list, x and y
	'''
	new_list2 = []
	for cur_car in car_list:
		if cur_car.tag == car_tag:
			new_car = Car(s = [int(to_position1), int(to_position2)],\
				l = int(cur_car.length), t = car_tag, \
				o = cur_car.orientation, p = cur_car.puzzle_tag)
		else:
			new_car = Car(s = [int(cur_car.start[0]), int(cur_car.start[1])], \
							l = int(cur_car.length), t = cur_car.tag,\
							o = cur_car.orientation, p = cur_car.puzzle_tag)
		new_list2.append(new_car)
		if new_car.tag == 'r':
			red = new_list2[-1]
	return new_list2, red

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
			
def is_solved(board, red): 
	'''
		return true if current board state can win
		no car in front of red
	'''
	cur_position = red.end[0] + 1 
	while(cur_position < board.width): # search right of red car
		if board.board_dict[(cur_position, red.start[1])] is not None:
			return False
		cur_position += 1
	return True

def assign_level(car_list): 
	''' 
	assign level to each car
	'''
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

def get_num_cars_from_levels(car_list, highest_level):
	''' 
		return the number of cars at each level 
		highest_level >= any possible min(cur_car.level)
		if a car level is > highest_level, ignore the car
	'''
	list_toreturn = [0] * (highest_level+1)
	for cur_car in car_list:
		if cur_car.level != []:
			minlevel = min(cur_car.level)
			if minlevel <= highest_level:
				list_toreturn[min(cur_car.level)] += 1
	return list_toreturn

def construct_mag(board, red):
	'''
		assign graph edges and neighbors, return a new car list
	'''
	queue = []
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
	
	# TODO should return a dictionary as a mag object


def extract_mag(board):
	"""
	should return a mag, which is a dict
	takes the board and understands the connections
	and returns it
	"""
	pass

def apply_mag(board, mag):
	"""
	takes the connections represented by mag (dict)
	applies them to the cars on the board
	for car in mag:
		board.car.edge_to = mag[car]
	if get rid of edge_to in Car, we do not need this function
	"""
	pass

def offline_generate_mags():
	"""
	for every puzzle, 
		generate all board configurations (try all positions and drop overlaps)
		for every configuration(board)
			call construct_mag(board)
			mag = extract_mag(board)
			store mag in a file under id of board
	"""
	pass











