import numpy
from kaggle_environments.envs.halite.helpers import Board
from typing import Tuple
import math


def distance(from_pos, to_pos):
    d = difference(from_pos, to_pos)
    return math.sqrt(d[0] ** 2 + d[1] ** 2)


def difference(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
    return to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]


class HaliteBoard:
    def __init__(self, board, config):
        # Square board
        self.size = config.size
        self.dims = (self.size, self.size)
        self.board_obj = Board(board, config)
        self.board = numpy.reshape(board, newshape=self.dims)

    def list_pos_to_board_pos(self, list_pos: int) -> Tuple[int, int]:
        """
        Convert 1d list position to 2d board position
        :param list_pos: 1d position in list
        """
        p = divmod(list_pos, self.size)
        return p[1], p[0]
