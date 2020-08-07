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

        self._ordered_player_ids = self.calculate_p_id_list()

        self.board = self.parse_board()

    def calculate_p_id_list(self):
        """
        Calculates the ordered player ID list. Used for parsing the board.
        :return: A list of all player IDs with the agent's player ID at index 0
        """
        ids = set(self.player_list.keys())
        ids.remove(self.current_player_id)
        return [self.current_player_id] + list(ids)

    def parse_board(self):
        out_array = np.zeros(self.dims)
        cells = self.cells
        for cell in cells.values():
            out_array[:, cell.position.y, cell.position.x] = self.parse_cell(cell)
        return out_array

    def parse_cell(self, cell: Cell):
        out = [0 for _ in range(self.dims[0])]
        # Bound halite between 0 and 1
        out[0] = cell.halite / self.settings["max_cell_halite"]

        if cell.ship is not None:
            ship_idx = self._ordered_player_ids.index(cell.ship.player_id) + 1
            out[ship_idx] = 1

        if cell.shipyard is not None:
            shipyard_idx = self._ordered_player_ids.index(cell.shipyard.player_id) + 5
            out[shipyard_idx] = 1

        return out

    def list_pos_to_board_pos(self, list_pos: int) -> Point:
        """
        Convert 1d list position to 2d board position
        :param list_pos: 1d position in list
        """
        p = divmod(list_pos, self.size)
        return Point(p[1], p[0])


if __name__ == "__main__":
    a = np.zeros((10, 10, 10))

    a[:, 1, 1] = [i for i in range(10)]
