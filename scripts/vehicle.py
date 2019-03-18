# rushhour cars class
CAR_IDS = {'X', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'}
TRUCK_IDS = {'O', 'P', 'Q', 'R'}
max_x = 5
max_y = 5

class Vehicle(object):
    """A configuration of a single vehicle."""

    def __init__(self, id, x, y, l, orientation):
        """Create a new vehicle.
        
        Arguments:
            id: a valid car or truck id character
            x: the x coordinate of the top left corner of the vehicle (0-5)
            y: the y coordinate of the top left corner of the vehicle (0-5)
            (board[y][x])
            l: the length of the car, typically value 2 or 3
            orientation: either the vehicle is vertical or horizontal

        Exceptions:
            ValueError: on invalid id, x, y, or orientation
        """
        # record if
        self.id = id
        # record length
        if 0 <= l <= max_x and 0 <= l <= max_y:
            self.length = l
        else:
            raise ValueError('Invalid id {0}'.format(id))
        # record x and y starting positions
        if 0 <= x <= max_x:
            self.x = x
        else:
            raise ValueError('Invalid x {0}'.format(x))
        if 0 <= y <= max_y:
            self.y = y
        else:
            raise ValueError('Invalid y {0}'.format(y))
        # x and y ending positions
        if orientation == 'horizontal':
            self.orientation = orientation
            x_end = self.x + (self.length - 1)
            y_end = self.y
        elif orientation == 'vertical':
            self.orientation = orientation
            x_end = self.x
            y_end = self.y + (self.length - 1)
        else:
            raise ValueError('Invalid orientation {0}'.format(orientation))
        if x_end > max_x or y_end > max_x:
            raise ValueError('Invalid configuration')

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "Vehicle({0}, {1}, {2}, {3}, {4})".format(self.id, self.x, self.y,
                                                    self.length, self.orientation)
