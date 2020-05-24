from Board import *

class MAG:
	def __init__(self, board):
		self.board = board
		self.car_levels = {} # stores level for each car
		self.neighbor_dict = {} # store neighbors of each car
		self.highest_level = 7 # highest mag level, red is level 0
		self.num_cars_each_level = None

	def construct(self):
		board = self.board
		queue = []
		visited = []
		self.num_cars_each_level = [0]*(self.highest_level+1)
		visited.append(board.board_dict[(board.redxy[0], board.redxy[1])])
		self.neighbor_dict[board.board_dict[(board.redxy[0], board.redxy[1])]] = []
		self.car_levels[board.board_dict[(board.redxy[0], board.redxy[1])]] = 0
		self.num_cars_each_level[0] += 1
		for x in range(board.redxy[0]+board.board_dict[board.redxy][1], board.width): # search red right
			y = board.redxy[1]
			meet_car = board.board_dict.get((x, y))
			if meet_car != None:
				queue.append(meet_car)
				visited.append(meet_car)
				self.neighbor_dict[board.board_dict[(board.redxy[0], board.redxy[1])]].append(meet_car)
				self.car_levels[meet_car] = 1
				self.num_cars_each_level[1] += 1
		while len(queue) != 0: # obstacle, Car: (start, length, tag, orientation)
			cur_car	= queue.pop(0)
			self.neighbor_dict[cur_car]=[]
			if cur_car[3] == 'vertical': # vertical
				x = cur_car[0][0]
				for y in range(cur_car[0][1]-1, -1, -1): # search up
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in self.neighbor_dict[cur_car]:
						self.neighbor_dict[cur_car].append(meet_car)
						if meet_car not in self.car_levels:
							self.car_levels[meet_car] = self.car_levels[cur_car]+1
							self.num_cars_each_level[self.car_levels[cur_car]+1] += 1
						if meet_car not in visited:
							queue.append(meet_car)
							visited.append(meet_car)
				for y in range(cur_car[0][1]+cur_car[1], board.height, 1): # search down
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in self.neighbor_dict[cur_car]:
						self.neighbor_dict[cur_car].append(meet_car)
						if meet_car not in self.car_levels:
							self.car_levels[meet_car] = self.car_levels[cur_car]+1
							self.num_cars_each_level[self.car_levels[cur_car]+1] += 1
						if meet_car not in visited:
							queue.append(meet_car)
							visited.append(meet_car)
			elif cur_car[3] == 'horizontal': # or horizontal
				y = cur_car[0][1]
				for x in range(cur_car[0][0]-1,-1,-1): # search left
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in self.neighbor_dict[cur_car]:
						self.neighbor_dict[cur_car].append(meet_car)
						if meet_car not in self.car_levels:
							self.car_levels[meet_car] = self.car_levels[cur_car]+1
							self.num_cars_each_level[self.car_levels[cur_car]+1] += 1
						if meet_car not in visited:
							queue.append(meet_car)
							visited.append(meet_car)
				for x in range(cur_car[0][0]+cur_car[1],board.width,1): #search right
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in self.neighbor_dict[cur_car]:
						self.neighbor_dict[cur_car].append(meet_car)
						if meet_car not in self.car_levels:
							self.car_levels[meet_car] = self.car_levels[cur_car]+1
							self.num_cars_each_level[self.car_levels[cur_car]+1] += 1
						if meet_car not in visited:
							queue.append(meet_car)
							visited.append(meet_car)
			visited.append(cur_car)
		return self.num_cars_each_level

	def easy_construct(self):
		board = self.board
		queue = []
		levels = []
		visited = []
		self.num_cars_each_level = [0]*(self.highest_level+1)
		visited.append(board.board_dict[(board.redxy[0], board.redxy[1])])
		self.num_cars_each_level[0] += 1
		for x in range(board.redxy[0]+board.board_dict[board.redxy][1], board.width): # search red right
			y = board.redxy[1]
			meet_car = board.board_dict.get((x, y))
			if meet_car != None:
				queue.append(meet_car)
				levels.append(1)
				visited.append(meet_car)
				self.num_cars_each_level[1] += 1
		while len(queue) != 0: # obstacle, Car: (start, length, tag, orientation)
			cur_car	= queue.pop(0)
			cur_level = levels.pop(0)
			if cur_car[3] == 'vertical': # vertical
				x = cur_car[0][0]
				for y in range(cur_car[0][1]-1, -1, -1): # search up
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in visited:
							self.num_cars_each_level[cur_level+1] += 1
							queue.append(meet_car)
							levels.append(cur_level+1)
							visited.append(meet_car)
				for y in range(cur_car[0][1]+cur_car[1], board.height, 1): # search down
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in visited:
							self.num_cars_each_level[cur_level+1] += 1
							queue.append(meet_car)
							levels.append(cur_level + 1)
							visited.append(meet_car)
			elif cur_car[3] == 'horizontal': # or horizontal
				y = cur_car[0][1]
				for x in range(cur_car[0][0]-1,-1,-1): # search left
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in visited:
							self.num_cars_each_level[cur_level+1] += 1
							queue.append(meet_car)
							levels.append(cur_level+1)
							visited.append(meet_car)
				for x in range(cur_car[0][0]+cur_car[1],board.width,1): #search right
					meet_car =  board.board_dict.get((x,y))
					if meet_car != None and meet_car not in visited:
							self.num_cars_each_level[cur_level+1] += 1
							queue.append(meet_car)
							levels.append(cur_level+1)
							visited.append(meet_car)
			visited.append(cur_car)
		return self.num_cars_each_level
	

# if __name__ == "__main__":
# 	inspath = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
# 	ins = random.choice(os.listdir(inspath))
# 	print(ins)
# 	board = json_to_board('/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+ins)
# 	board.print_board()
# 	for b in all_legal_moves(board):
# 		print('------------- new legal board')
# 		b.print_board()
# 		print(is_solved(b))
# 		mag = MAG()
# 		print(mag.easy_construct(b))
# 		# for (key, value) in mag.neighbor_dict.items():
# 		# 	print(key, value)
# 	board.print_board()

