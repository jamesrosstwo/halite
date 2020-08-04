import numpy
from kaggle_environments.envs.halite.helpers import Board, Configuration, Point
from typing import Tuple, Dict, Any, Union
import math


def pos_distance(from_pos: Point, to_pos: Point) -> float:
    d = pos_difference(from_pos, to_pos)
    return math.sqrt(d[0] ** 2 + d[1] ** 2)


def pos_difference(from_pos: Point, to_pos: Point) -> Point:
    return Point(to_pos.x - from_pos.x, to_pos.y - from_pos.y)


class HaliteBoard:
    def __init__(self, board: Dict[str, Any], config: Union[Configuration, Dict[str, Any]]):
        # Square board
        self.size = config.size
        self.dims = (self.size, self.size)
        self.board_obj = Board(board, config)
        self.board = numpy.reshape(board["halite"], newshape=self.dims)

    def list_pos_to_board_pos(self, list_pos: int) -> Point:
        """
        Convert 1d list position to 2d board position
        :param list_pos: 1d position in list
        """
        p = divmod(list_pos, self.size)
        return Point(p[1], p[0])
