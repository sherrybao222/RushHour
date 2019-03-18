# rushhour solution generator using BFS, print out many solutions
# not optimal!!!!!!!!!!
# need to run with python2.7
# associated with file vehicle.py, and difficulty.py

import sys
from collections import deque
from vehicle import Vehicle
import MAG
import numpy as np
import json

GOAL_VEHICLE = Vehicle('r', 4, 2, 2, 'horizontal') # nameX, x4, y2, l2, horizontal
max_x = 5 # maximum x coordinate value
max_y = 5 # maximum y coordinate value

class RushHour(object):
    """A configuration of a single Rush Hour board."""

    def __init__(self, vehicles):
        """Create a new Rush Hour board.
        
        Arguments:
            vehicles: a set of Vehicle objects.
        """
        self.vehicles = vehicles

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.vehicles == other.vehicles

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        s = '-' * 8 + '\n'
        for line in self.get_board():
            s += '|{0}|\n'.format(''.join(line))
        s += '-' * 8 + '\n'
        return s

    def get_board(self):
        """Representation of the Rush Hour board as a 2D list of strings"""
        board = [[' ', ' ', ' ', ' ', ' ', ' '],
                 [' ', ' ', ' ', ' ', ' ', ' '],
                 [' ', ' ', ' ', ' ', ' ', ' '],
                 [' ', ' ', ' ', ' ', ' ', ' '],
                 [' ', ' ', ' ', ' ', ' ', ' '],
                 [' ', ' ', ' ', ' ', ' ', ' ']]
        for vehicle in self.vehicles:
            x, y = vehicle.x, vehicle.y
            if vehicle.orientation == 'horizontal':
                for i in range(vehicle.length):
                    board[y][x+i] = vehicle.id
            else:
                for i in range(vehicle.length):
                    board[y+i][x] = vehicle.id
        return board

    def solved(self):
        """Returns true if the board is in a solved state."""
        return GOAL_VEHICLE in self.vehicles

    def moves(self):
        """Return iterator of next possible moves."""
        board = self.get_board()
        for v in self.vehicles:
            if v.orientation == 'horizontal':
                if v.x - 1 >= 0 and board[v.y][v.x - 1] == ' ': # move left
                    new_v = Vehicle(v.id, v.x - 1, v.y, v.length, v.orientation)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles.remove(v)
                    new_vehicles.add(new_v)
                    yield RushHour(new_vehicles)
                if v.x + v.length <= max_x and board[v.y][v.x + v.length] == ' ': # move right
                    new_v = Vehicle(v.id, v.x + 1, v.y, v.length, v.orientation)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles.remove(v)
                    new_vehicles.add(new_v)
                    yield RushHour(new_vehicles)
            else:
                if v.y - 1 >= 0 and board[v.y - 1][v.x] == ' ': # move up
                    new_v = Vehicle(v.id, v.x, v.y - 1, v.length, v.orientation)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles.remove(v)
                    new_vehicles.add(new_v)
                    yield RushHour(new_vehicles)
                if v.y + v.length <= max_y and board[v.y + v.length][v.x] == ' ': # move down
                    new_v = Vehicle(v.id, v.x, v.y + 1, v.length, v.orientation)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles.remove(v)
                    new_vehicles.add(new_v)
                    yield RushHour(new_vehicles)


# shift a list
def shift(l, n):
    return l[n:] + l[:n]

# json file to RushHour vehicles class
def json_to_car_list(filename):
    with open(filename,'r') as data_file:
        vehicles = []
        data = json.load(data_file)
        for c in data['cars']:
            cur_car = Vehicle(id = c['id'], x = int(c['position'])%6, y = int(c['position']/6), \
                l = int(c['length']), orientation = c['orientation'])
            vehicles.append(cur_car)
            print("new car : " + cur_car.id + " x: " + str(cur_car.x) \
                + " y: "  + str(cur_car.y) \
                + " length: " + str(cur_car.length) \
                + " orientation: " + str(cur_car.orientation))
    return RushHour(set(vehicles))

# def load_file(rushhour_file):
#     vehicles = []
#     for line in rushhour_file:
#         line = line[:-1] if line.endswith('\n') else line
#         id, x, y, l, orientation = line
#         vehicles.append(Vehicle(id, int(x), int(y), int(l), orientation))
#     return RushHour(set(vehicles))

def breadth_first_search(r, shift, max_depth=25):
    """
    Find solutions to given RushHour board using breadth first search.
    Returns a dictionary with named fields:
        visited: the number of configurations visited in the search
        solutions: paths to the goal state
        depth_states: the number of states visited at each depth

    Arguments:
        r: A RushHour board.

    Keyword Arguments:
        max_depth: Maximum depth to traverse in search (default=25)
    """
    visited = set()
    solutions = list()
    depth_states = dict()

    queue = deque()
    queue.appendleft((r, tuple()))
    i = 0
    while len(queue) != 0:
        board, path = queue.pop()
        # print board 
        # print path
        
        new_path = path + tuple([board])

        depth_states[len(new_path)] = depth_states.get(len(new_path), 0) + 1

        if len(new_path) >= max_depth:
            break

        if board in visited:
            continue
        else:
            visited.add(board)

        if board.solved():
            solutions.append(new_path)
        else:
            # all_moves = board.moves()
            # for move in board.moves():
                # queue.extendleft((move, new_path))
            queue.extendleft((move, new_path) for move in board.moves())
            
            if i == 0:
                first_qlen = len(queue)
                for j in range(0, shift):
                    queue.append(queue.popleft())
            # print queue
            # sys.exit()

        i += 1

    return {'visited': visited,
            'solutions': solutions,
            'depth_states': depth_states}, first_qlen

def print_solution_steps(solution):
    """Generate list of steps from a solution path."""
    steps = []
    for i in range(len(solution) - 1):
        r1, r2 = solution[i], solution[i+1]
        v1 = list(r1.vehicles - r2.vehicles)[0]
        v2 = list(r2.vehicles - r1.vehicles)[0]
        if v1.x < v2.x:
            steps.append('{0}R'.format(v1.id))
        elif v1.x > v2.x:
            steps.append('{0}L'.format(v1.id))
        elif v1.y < v2.y:
            steps.append('{0}D'.format(v1.id))
        elif v1.y > v2.y:
            steps.append('{0}U'.format(v1.id))
    return steps

def condense_solution(solution_steps):
    '''Condense solution to minimal number of moves'''
    new_solution_steps = []
    prev_id = ''
    step = 0
    for fast in range(len(solution_steps)):
        cur_id = solution_steps[fast][0]
        if cur_id == prev_id:
            if solution_steps[fast][1] == 'U' or solution_steps[fast][1] == 'L':
                step -= 1
            elif solution_steps[fast][1] == 'D' or solution_steps[fast][1] == 'R':
                step += 1
        else:
            if fast > 0:
                new_solution_steps.append(str(prev_id) + ' ' + str(step))
            step = 0
            if solution_steps[fast][1] == 'U' or solution_steps[fast][1] == 'L':
                step -= 1
            elif solution_steps[fast][1] == 'D' or solution_steps[fast][1] == 'R':
                step += 1
        
        if fast == len(solution_steps) - 1:
            new_solution_steps.append(str(cur_id) + ' ' + str(step))
        
        prev_id = cur_id

    return new_solution_steps



if __name__ == '__main__':
    filename = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/prb6671.json'
    rushhour = json_to_car_list(filename)

    results, qlen = breadth_first_search(rushhour, shift = 0, max_depth=150)
    # print qlen # 9: valid 1-8
    print len(results['solutions'])
    all_solution_steps = set()
    for solution in results['solutions']:
        all_solution_steps.add(', '.join(print_solution_steps(solution)))

    for n in range(1, qlen):
        results, _ = breadth_first_search(rushhour, shift = n, max_depth=150)
        print len(results['solutions'])
        for solution in results['solutions']:
            all_solution_steps.add(', '.join(print_solution_steps(solution)))

    # print all_solution_steps
    print len(all_solution_steps)
    condensed_solution_steps = []
    optlen = float('inf')
    for steps in all_solution_steps:
        c = condense_solution(list(steps.split(', ')))
        condensed_solution_steps.append(c)
        if len(c) < optlen:
            optlen = len(c)
        # print list(steps.split(', '))
        print condensed_solution_steps[-1]

    print('optlen: ', optlen)

    # save only optlen solutions
    opt_solutions = []
    for c in condensed_solution_steps:
        if len(c) == optlen:
            opt_solutions.append(c)
            print c
    print('number of opt solutions found ', len(opt_solutions))

    # condensed_solutions = condense_solution_steps(results['solutions'])
    # solution_steps = []
    # for solution in results['solutions']:
    #     solution_steps.append(', '.join(print_solution_steps(solution)))
    #     print('Solution number ' + str(len(solution_steps)))
    #     print(solution_steps[-1])


    # print('{0} Solutions found'.format(len(solution_steps)))
    # for solution in results['solutions']:
    #     print 'Solution: {0}'.format(', '.join(print_solution_steps(solution)))

    print '{0} Nodes visited'.format(len(results['visited']))
