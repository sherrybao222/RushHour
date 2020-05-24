'''
Car: (start(x,y), length, tag, orientation)
'''
from json import dump, load
from itertools import product
import os, random, copy, pickle, sys
import numpy as np

class Board:
	def __init__(self):
		self.height = 6
		self.width = 6
		self.board_dict = {}
		self.redxy = None
		self.tag_dict = {}
		self.id = ''
		self.children_id = []
	def print_board(self):
		out_str = ''
		for y in range(self.height):
			for x in range(self.width):
				if self.board_dict.get((x,y)) == None:
					out_str += '.'
					if y == 2 and x == 5:
						out_str += '>'
					continue
				elif self.board_dict[(x,y)][2] == 'r':
					out_str += 'R'
				else:
					out_str += self.board_dict[(x,y)][2]
				if y == 2 and x == 5:
					out_str += '>'
			out_str += '\n'
		print(out_str)
		return out_str

def json_to_board(filename):
	with open(filename, 'r') as datafile:
		board = Board()
		data = load(datafile)
		for c in data['cars']:
			car = ((int(c['position'])%6, int(c['position']/6)),
				int(c['length']), str(c['id']), c['orientation'])
			board.tag_dict[car[2]] = car
			if car[2] == 'r':
				board.redxy = car[0]
			if car[3] == 'horizontal':
				for x in range(car[1]):
					board.board_dict[(car[0][0]+x, car[0][1])] = car
			elif car[3] == 'vertical':
				for y in range(car[1]):
					board.board_dict[(car[0][0], car[0][1]+y)] = car
	return board


def move_xy(board, tag, tox, toy):
	tox = int(tox)
	toy = int(toy)
	newboard = Board()
	for x in range(newboard.width):
		for y in range(newboard.height):
			car = board.board_dict.get((x,y))
			if car != None:
				newboard.board_dict[(x,y)]=car
				newboard.tag_dict[car[2]]=car
	newboard.redxy = board.redxy
	carinfo = board.tag_dict[tag]
	newcar = ((int(tox), int(toy)), carinfo[1], carinfo[2], carinfo[3])
	newboard.tag_dict[tag] = newcar
	if carinfo[3]=='horizontal':
		for x in range(carinfo[1]):
			newboard.board_dict[(carinfo[0][0]+x, carinfo[0][1])] = None
		for x in range(carinfo[1]):
			newboard.board_dict[(tox+x, toy)] = newcar
	elif carinfo[3]=='vertical':
		for y in range(carinfo[1]):
			newboard.board_dict[(carinfo[0][0], carinfo[0][1]+y)] = None
		for y in range(carinfo[1]):
			newboard.board_dict[(tox, toy+y)] = newcar
	if carinfo[2]=='r':
		newboard.redxy = (tox, toy)
	return newboard


def move(board, tag, toxy): 
	'''
		make a move and return the new car list, 
		single position label
	'''
	return move_xy(board, tag, int(toxy%6), int(toxy/6))


def all_legal_moves(board):
	visited_cars = []
	all_legal_boards = []
	for tag in board.tag_dict:
		curcar = board.tag_dict[tag]
		visited_cars.append(curcar)
		if curcar[3] == 'horizontal':
			leftx = curcar[0][0]-1
			lefty = curcar[0][1]
			while(leftx >= 0 and board.board_dict.get((leftx, lefty)) == None):
				all_legal_boards.append(move_xy(board, curcar[2], leftx, lefty))
				leftx -= 1
			rightx = curcar[0][0]+curcar[1] # search right
			righty = curcar[0][1]
			while(rightx < board.width and board.board_dict.get((rightx, righty)) == None):
				all_legal_boards.append(move_xy(board, curcar[2], rightx-curcar[1]+1, righty))
				rightx += 1
		elif curcar[3] == 'vertical':
			upx = curcar[0][0]
			upy = curcar[0][1] - 1 # search up
			while(upy >= 0 and board.board_dict.get((upx, upy)) == None):
				all_legal_boards.append(move_xy(board, curcar[2], upx, upy))
				upy -= 1
			downx = curcar[0][0]
			downy = curcar[0][1]+curcar[1] # searc down
			while(downy < board.height and board.board_dict.get((downx, downy)) == None):
				all_legal_boards.append(move_xy(board, curcar[2], downx, downy-curcar[1]+1))
				downy += 1
	return all_legal_boards


def is_solved(board):
	''' returns true if current board position is a winning position '''
	for x in range(board.redxy[0]+board.board_dict[board.redxy][1], board.width):
		if board.board_dict.get((x,board.redxy[1])) != None:
			return False
	return True


def id_to_board(self, idstr, puzzle, puzzle_info_path='/Users/chloe/Documents/RushHour/exp_data/preprocessed_positions/'):
	'''
	TODO: convert board id string to a real board, puzzle info (car orientation) is stored in puzzle_path
	tag, len, startx, starty (, vertical1 information saved in separate file) 
	'''
	pass

def make_id(board):
	''' 
	tag, len, startx, starty (, vertical1 information saved in separate file) 
	'''
	out = ''
	for tag in sorted(board.tag_dict): # sort tag keys 
		car = board.tag_dict[tag]
		out += str(tag)
		out += str(car[1])
		out += str(car[0][0])
		out += str(car[0][1])
	board.id = out
	return out

def pickle_board(board, path='/Users/chloe/Documents/RushHour/exp_data/preprocessed_positions/'):
	pickle.dump(board, open(path+board.id+'.p', 'wb'))
	return path+board.id+'.p'

def load_board(board_id, path='/Users/chloe/Documents/RushHour/exp_data/preprocessed_positions/'):
	return pickle.load(open(path+board_id+'.p', 'rb'))

def generate_all_positions_of_car(tag, board):
	'''
	generate all possible positions of a car on the given board
	return a list of car tuples
	'''
	car = board.tag_dict[tag] # (start(x,y), length, tag, orientation)
	result = []
	if car[3]=='vertical':
		for y in range(0, board.height-car[1]+1):
			result.append(((car[0][0], y), car[1], tag, 'vertical'))
	elif car[3]=='horizontal':
		for x in range(0,board.width-car[1]+1):
			result.append(((x,car[0][1]), car[1], tag, 'horizontal'))
	return result

def board_position_is_legal(board_position, hash_x, hash_y):
	'''
	check if board position is legal / no overlap
	return true if legal, false if illegal
	'''
	hash_board = np.zeros(shape=(hash_y, hash_x), dtype=int) #[y][x]
	for car in board_position: # (start(x,y), length, tag, orientation)
		if car[3] == 'vertical':
			for y in range(car[0][1], car[0][1]+car[1]):
				if hash_board[y][car[0][0]] == 1:
					return False
				hash_board[y][car[0][0]] = 1
		elif car[3] == 'horizontal':
			for x in range(car[0][0], car[0][0]+car[1]):
				if hash_board[car[0][1]][x] == 1:
					return False
				hash_board[car[0][1]][x] = 1
	return True


def drop_illegal_positions(all_board_positions, initial_board):
	'''
	drop the positions with overlapping cars
	return a list of positions
	each position is a list of car tuples
	'''
	result = []
	for board_position in all_board_positions:
		if not board_position_is_legal(board_position, initial_board.width, initial_board.height):
			continue
		result.append(board_position)
	return result


def generate_all_positions_of_board(board):
	'''
	generate all possible positions from a board
	return a list of positions
	each position contains a list of car tuples
	'''
	positions_for_each_car = [generate_all_positions_of_car(tag, board) for tag in sorted(board.tag_dict)]
	all_board_positions = product(*positions_for_each_car)
	return drop_illegal_positions(all_board_positions, board)


def position_to_board(position):
	'''
	convert position to board
	position is an array of cars on a given board
	return a board instance
	'''
	board = Board()
	for car in position:
		board.tag_dict[car[2]] = car
		if car[2] == 'r':
			board.redxy = car[0]
		if car[3] == 'horizontal':
			for x in range(car[1]):
				board.board_dict[(car[0][0]+x, car[0][1])] = car
		elif car[3] == 'vertical':
			for y in range(car[1]):
				board.board_dict[(car[0][0], car[0][1]+y)] = car
	return board

def create_puzzle_info_file(puzzle_id, puzzle_dir='/Users/chloe/Documents/RushHour/exp_data/data_adopted/', 
										out_dir='/Users/chloe/Documents/RushHour/exp_data/preprocessed_positions/'):
	''' create puzzle info dictionary 
		and save to pickle file
	'''
	data_dir = puzzle_dir + puzzle_id + '.json'
	board = json_to_board(data_dir)
	puzzle_dict = {}
	puzzle_dict['board'] = board
	for tag in board.tag_dict:
		car = board.tag_dict[tag]
		puzzle_dict[tag] = car
	pickle.dump(puzzle_dict, open(out_dir+puzzle_id+'/'+puzzle_id+'.p', 'wb'))
	return puzzle_dict


if __name__ == "__main__":
	inspath = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins = random.choice(os.listdir(inspath))
	print(ins)
	board = json_to_board('/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+ins)
	print(make_id(board))
	
	all_pos = generate_all_positions_of_board(board)

	for pos in all_pos:
		print('------------ new legal move')
		b = position_to_board(pos)
		print(make_id(b))
		b.print_board()

	print('--------original board')
	board.print_board()
	print(len(list(all_pos)))
		





