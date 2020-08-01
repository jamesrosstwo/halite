# Imports helper functions
from kaggle_environments.envs.halite.helpers import ShipAction

class HaliteShip:
    """
    Agent for an individual ship. Contains things like pathfinding to desired location,
    and performing ship actions
    """
    def __init__(self, pos):
        self.pos = pos

    def convert(self):
        pass

    def move_to(self, loc):
        pass

    # Returns best direction to move from one position (fromPos) to another (toPos)
    # Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
    def get_dir_to(self, toPos, size):
        fromX, fromY = divmod(self.pos[0], size), divmod(self.pos[1], size)
        toX, toY = divmod(toPos[0], size), divmod(toPos[1], size)
        if fromY < toY: return ShipAction.NORTH
        if fromY > toY: return ShipAction.SOUTH
        if fromX < toX: return ShipAction.EAST
        if fromX > toX: return ShipAction.WEST