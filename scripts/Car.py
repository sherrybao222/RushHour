class Car:
	def __init__(self, s, l, t, o, p):
		self.start = s # [hor, ver]
		self.length = l # int
		self.tag = t # str
		self.orientation = o # str
		self.puzzle_tag = p # str
		if self.orientation == 'horizontal':
			self.end = [self.start[0] + self.length - 1, self.start[1]]
		elif self.orientation == 'vertical':
			self.end = [self.start[0], self.start[1] + self.length - 1]
		self.edge_to = []
		self.level = []
		self.visited = False
