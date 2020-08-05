# Imports helper functions
from enum import Enum, auto
from typing import Optional

from src.board.board import pos_difference

from kaggle_environments.envs.halite.helpers import Ship, ShipAction, Point, ShipId, PlayerId, Board

# Directions a ship can move
SHIP_DIRECTIONS = (None, ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST)


class HaliteShipState(Enum):
    COLLECT = auto()
    DEPOSIT = auto()


class HaliteShip(Ship):
    """
    Agent for an individual ship. Contains things like pathfinding to desired location,
    and performing ship actions
    """

    def __init__(self, ship_id: ShipId, position: Point, halite: int, player_id: PlayerId, board: 'Board'):
        super().__init__(ship_id, position, halite, player_id, board)
        self.state = HaliteShipState.COLLECT

    @classmethod
    def from_ship(cls, ship_obj: Ship):
        return cls(ship_obj.id, ship_obj.position, ship_obj.halite, ship_obj.player_id, ship_obj._board)

    def convert(self):
        self.next_action = ShipAction.CONVERT

    def move_to(self, loc: Point):
        self.next_action = self.get_dir_to(loc)

    def move_dir(self, dir: ShipAction):
        self.next_action = dir

    def adjacent_halite_counts(self):
        return self.cell.halite, \
               self.cell.north.halite, \
               self.cell.east.halite, \
               self.cell.south.halite, \
               self.cell.west.halite,

    def max_neighbouring_halite_dir(self):
        counts = self.adjacent_halite_counts()
        best = max(range(len(counts)), key=counts.__getitem__)
        return SHIP_DIRECTIONS[best]

    def move_to_max_adjacent_halite(self):
        self.next_action = self.max_neighbouring_halite_dir()

    # Returns best direction to move from one position (fromPos) to another (toPos)
    # Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
    def get_dir_to(self, to_pos: Point) -> Optional[ShipAction]:
        pos_diff = pos_difference(self.position, to_pos)

        action = None

        if abs(pos_diff[0]) > abs(pos_diff[1]):
            action = ShipAction.EAST if pos_diff[0] > 0 else ShipAction.WEST
        elif pos_diff[1] > 0:
            action = ShipAction.NORTH
        elif pos_diff[1] < 0:
            action = ShipAction.SOUTH
        return action
