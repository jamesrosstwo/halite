from kaggle_environments.envs.halite.helpers import Board, Configuration, Point, Cell
from typing import Dict, Any, Union
import numpy as np
from src.constants import SETTINGS


def pos_distance(from_pos: Point, to_pos: Point) -> float:
    d = pos_difference(from_pos, to_pos)
    return sum(map(abs, d))


def pos_difference(from_pos: Point, to_pos: Point) -> Point:
    return Point(to_pos.x - from_pos.x, to_pos.y - from_pos.y)


class HaliteBoard(Board):
    def __init__(self, board: Dict[str, Any], config: Union[Configuration, Dict[str, Any]]):
        from src.entities.player import HalitePlayer
        # Square board
        super().__init__(board, config)
        self.settings = SETTINGS["board"]
        self.size = config.size
        self.dims = tuple(self.settings["size"])
        self.player_list = {id: HalitePlayer.from_player(p) for id, p in self.players.items()}
        self.player = self.player_list[self.current_player_id]
        self.board = self.parse_board()


    def parse_board(self):
        out_array = np.zeros(self.dims)
        cells = self.cells
        for cell in cells.values():
            out_array[:, cell.position.y, cell.position.x] = self.parse_cell(cell)
        return out_array

    def parse_cell(self, cell: Cell):
        out = [0 for x in range(self.dims[2])]
        # Bound halite between 0 and 1
        out[0] = cell.halite / self.settings["max_cell_halite"]



    def list_pos_to_board_pos(self, list_pos: int) -> Point:
        """
        Convert 1d list position to 2d board position
        :param list_pos: 1d position in list
        """
        p = divmod(list_pos, self.size)
        return Point(p[1], p[0])


if __name__ == "__main__":
    a = np.zeros((10, 10, 10))

    a[:,1,1] = [i for i in range(10)]
