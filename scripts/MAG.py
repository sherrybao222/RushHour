class Car:
	start, end, length = [0,0], [0,0], [0,0] #hor,ver
	tag = ''
	puzzle_tag = ''
	orientation = ''
	edge_to = []
	def __init__(self, s, l, t, o, p):
		self.start = s
		self.length = l
		self.tag = t
		self.orientation = o
		self.puzzle_tag = p
		if orientation == 'horizental':
			self.end = [self.start[0] + self.length - 1, self.start[1]]
		else:
			self.end = [self.start[0], self.start[1] + self.length - 1]
	def add_edge_to(self, c): # point to car c
		self.edge_to.add(c)
	def remove_edge_to(self, c):
		if len(edge_to) == 0:
			return -1
		else:
			self.remove(c)

class Board:
	height, width = 6, 6
	board_dict = {}
	puzzle_tag = ''
	def __init__(self):
		for i in range(0, height):
			for j in range(0, width):
				self.board_dict[[i, j]: None]


def json_to_car_list(file_name):
	car_list = []
    with open(filename,'r') as data_file:
    data = load(data_file)
    for c in data['cars']:
        cur_car = Car(s = [int(c['position'])%6, int(c['position']/6)], \
        	l = int(c['length']), t = c['id'], o = c['orientation'], p = data['id'])
        car_list.append(cur_car)
	return car_list

def construct_board(self, car_list):
	board = Board()
	for car in car_list:
		cur_start = car.start
		cur_len = car.length
		cur_orientation = car.orientation
		occupied_space = []
		if cur_orientation == 'horizental':
			for i in range(0, car_len):
				occupied_space.append([cur_start[0] + i, cur_start[1]])
		else:
			for i in range(0, car_len):
				occupied_space.append([cur_start[0], cur_start[1] + i])
		for j in range(0, len(occupied_space)):
			board[occupied_space[j]:car]
	return board

def construct_mag(self, board, red):
	red_end = red.end
	stack = []
	for i in range(red_end[0] + 1, board.width - 1): #in front of red
		cur_car = board.board_dict[[i, red_end[1]]]
		if cur_car not None: #exists on board
			if cur_car not in red.edge_to:
				red.add_edge_to(cur_car)
				stack.append(cur_car)
	while len(stack) != 0:
		cur_car	= stack.pop()
		if cur_car.orientation == 'vertical':
			for j in range(cur_car.start[1] - 1, 0): # upper
				meet_car =  board.board_dict[[cur_car.start[0], j]]
				if meet_car not None:
					if meet_car not in cur_car.edge_to:
						cur_car.add_edge_to(meet_car)
						stack.append(meet_car)
			for k in range(cur_car.end[1] + 1, board.length - 1): # lower
				meet_car =  board.board_dict[[cur_car.start[0], j]]
				if meet_car not None:
					if meet_car not in cur_car.edge_to:
						cur_car.add_edge_to(meet_car)
						stack.append(meet_car)
		if cur_car.orientation == 'horizontal':
			for j in range(cur_car.start - 1, 0): # upper
				meet_car =  board.board_dict[[cur_car.start[0], j]]
				if meet_car not None:
					if meet_car not in cur_car.edge_to:
						cur_car.add_edge_to(meet_car)
						stack.append(meet_car)
			for k in range(cur_car.end + 1, board.length - 1): # lower
				meet_car =  board.board_dict[[cur_car.start[0], j]]
				if meet_car not None:
					if meet_car not in cur_car.edge_to:
						cur_car.add_edge_to(meet_car)
						stack.append(meet_car)

	