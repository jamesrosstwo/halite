from kaggle_environments.envs.halite.helpers import Board, Configuration, Point, Cell, PlayerId, ShipId, ShipyardId
from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
from src.constants import SETTINGS, TORCH_DEVICE
import torch



def pos_distance(from_pos: Point, to_pos: Point) -> float:
    d = pos_difference(from_pos, to_pos)
    return sum(map(abs, d))


def pos_difference(from_pos: Point, to_pos: Point) -> Point:
    return Point(to_pos.x - from_pos.x, to_pos.y - from_pos.y)


def pos_from_indices(indices: Tuple[int, int]):
    return Point(indices[0], indices[1])


class HaliteBoard(Board):
    def __init__(self, board: Dict[str, Any], config: Union[Configuration, Dict[str, Any]]):
        # Square board
        super().__init__(board, config)

        self._halite_players = None
        self._halite_ships = None
        self._halite_shipyards = None
        self._halite_opponents = None
        self._halite_opponent_ships = None
        self._halite_opponent_shipyards = None

        self._populate_halite_objs()

        self.settings = SETTINGS["board"]
        self.size = config.size
        self.dims = tuple(self.settings["dims"])

        self._ordered_player_map = self.calculate_p_id_map()

        self.map = self.parse_map()
        self.additional_vals = self.get_additional_board_vals()

    @classmethod
    def from_board(cls, board: Board):
        return cls(board.observation, board.configuration)

    def _populate_halite_objs(self):
        self._halite_players = {p_id: HalitePlayer.from_player(p, self) for p_id, p in super().players.items()}
        self._halite_ships = {k: HaliteShip.from_ship(v, self) for k, v in super().ships.items()}
        self._halite_shipyards = {k: HaliteShipyard.from_shipyard(v, self) for k, v in super().shipyards.items()}
        self._halite_opponents = {p_id: p for p_id, p in self._halite_players.items() if p_id != self.current_player_id}
        self._halite_opponent_ships = list(np.asarray([p.ships for p in self._halite_opponents.values()]).flatten())
        self._halite_opponent_ships = list(np.asarray([p.shipyards for p in self._halite_opponents.values()]).flatten())

        for player in self._halite_players.values():
            player.set_halite_objs()

    @property
    def sorted_player_ids(self) -> List[int]:
        """
        The ordered player ID list. Used for parsing the board.
        :return: A list of all player IDs with the agent's player ID at index 0
        """
        return list(self._ordered_player_map.keys())

    def calculate_p_id_map(self) -> Dict[int, int]:

        ids = set(self.players.keys())
        ids.remove(self.current_player_id)
        id_list = [self.current_player_id] + list(ids)
        return {v: idx for idx, v in enumerate(id_list)}

    def parse_map(self) -> np.ndarray:
        out_array = np.zeros(self.dims, dtype=np.float32)
        cells = self.cells
        for cell in cells.values():
            out_array[:, cell.position.y, cell.position.x] = self.parse_cell(cell)
        return out_array

    def parse_cell(self, cell: Cell) -> List[float]:
        out = [0. for _ in range(self.dims[0])]
        # Bound halite between 0 and 1
        out[0] = cell.halite / self.settings["max_cell_halite"]

        if cell.ship is not None:
            ship_idx = self._ordered_player_map[cell.ship.player_id] + 1
            out[ship_idx] = 1.

        if cell.shipyard is not None:
            shipyard_idx = self._ordered_player_map[cell.shipyard.player_id] + 5
            out[shipyard_idx] = 1.

        return out

    def ship_map(self, player_id) -> np.ndarray:
        return self.map[player_id + 1, :, :]

    def get_additional_board_vals(self):
        additional_board_vals = [float(self.step) / 400]

        player_halite = [self.players[x].halite for x in self.sorted_player_ids]
        p_max_halite = max(player_halite)
        player_halite = [float(x) / p_max_halite for x in player_halite]

        additional_board_vals += player_halite
        return np.array(additional_board_vals)

    def ship_at_pos(self, pos: Point) -> Optional['HaliteShip']:
        cell = self.cells[pos]
        ship = cell.ship
        if ship is not None:
            return HaliteShip.from_ship(ship, self)

    def shipyard_at_pos(self, pos: Point) -> Optional['HaliteShipyard']:
        cell = self.cells[pos]
        shipyard = cell.shipyard
        if shipyard is not None:
            return HaliteShipyard.from_shipyard(shipyard, self)

    def shipyard_map(self, player_id) -> np.ndarray:
        return self.map[player_id + 5, :, :]

    def list_pos_to_board_pos(self, list_pos: int) -> Point:
        """
        Convert 1d list position to 2d board position
        :param list_pos: 1d position in list
        """
        p = divmod(list_pos, self.size)
        return Point(p[1], p[0])

    @property
    def halite_players(self) -> Dict[PlayerId, 'HalitePlayer']:
        return self._halite_players

    @property
    def player(self) -> 'HalitePlayer':
        return self.halite_players[self.current_player_id]

    @property
    def ships(self) -> Dict[ShipId, 'HaliteShip']:
        return self._halite_ships

    @property
    def shipyards(self) -> Dict[ShipyardId, 'HaliteShipyard']:
        return self._halite_shipyards

    @property
    def opponents(self) -> Dict[PlayerId, 'HalitePlayer']:
        return self._halite_opponents

    @property
    def opponent_ships(self) -> List['HaliteShip']:
        return self._halite_opponent_ships

    @property
    def opponent_shipyards(self) -> List['HaliteShipyard']:
        return self._halite_opponent_shipyards


from src.agent.entities.halite_ship import HaliteShip
from src.agent.entities.halite_shipyard import HaliteShipyard
from src.agent.entities.player import HalitePlayer

if __name__ == "__main__":
    a = np.zeros((10, 10, 10))

    a[:, 1, 1] = [i for i in range(10)]
