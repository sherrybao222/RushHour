# determines the difficulty level of rushhour game configuration
# need to run with python2.7

'''
This script responds with Easy, Moderate, or Hard, 
depending on the difficulty of the game. 
Additionally, if the game cannot be solved, 
the script responds with Impossible.
The difficulty of the game is determined by analyzing both 
the shortest solution to the game, and the number of possible solutions. 
A game can be easy either by having a very short solution path, 
or a large number of possible solutions. 
Conversely, a problem can be hard by having a very long shortest solution path, 
or having a small number of possible solutions.
'''

import sys
import rushhour

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename) as rushhour_file:
        r = rushhour.load_file(rushhour_file)

    results = rushhour.breadth_first_search(r, max_depth=100)
    solutions = results['solutions']
    num_solutions = len(solutions)

    if num_solutions == 0:
        print 'Impossible'
        sys.exit(1)

    solutions.sort(key=lambda x: len(x))
    shortest_solution = len(solutions[0])

    if shortest_solution < 20 or num_solutions > 200:
        print 'Easy'
    elif shortest_solution > 50 or num_solutions < 20:
        print 'Hard'
    else:
        print 'Moderate'
