from kaggle_environments.envs.halite.helpers import Board, Configuration, Point, Cell, PlayerId, ShipId, ShipyardId
from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
from src.constants import SETTINGS


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
        self.settings = SETTINGS["board"]
        self.size = config.size
        self.dims = tuple(self.settings["size"])
        self._ordered_player_ids = self.calculate_p_id_list()
        self.map = self.parse_map()

    def calculate_p_id_list(self) -> List[int]:
        """
        Calculates the ordered player ID list. Used for parsing the board.
        :return: A list of all player IDs with the agent's player ID at index 0
        """
        ids = set(self.players.keys())
        ids.remove(self.current_player_id)
        return [self.current_player_id] + list(ids)

    def parse_map(self) -> np.ndarray:
        out_array = np.zeros(self.dims)
        cells = self.cells
        for cell in cells.values():
            out_array[:, cell.position.y, cell.position.x] = self.parse_cell(cell)
        return out_array

    def parse_cell(self, cell: Cell) -> List[float]:
        out = [0. for _ in range(self.dims[0])]
        # Bound agent between 0 and 1
        out[0] = cell.halite / self.settings["max_cell_halite"]

        if cell.ship is not None:
            ship_idx = self._ordered_player_ids.index(cell.ship.player_id) + 1
            out[ship_idx] = 1.

        if cell.shipyard is not None:
            shipyard_idx = self._ordered_player_ids.index(cell.shipyard.player_id) + 5
            out[shipyard_idx] = 1.

        return out

    def ship_map(self, player_id) -> np.ndarray:
        return self.map[player_id + 1, :, :]

    def ship_at_pos(self, pos: Point) -> Optional['HaliteShip']:
        cell = self.cells[pos]
        ship = cell.ship
        if ship is not None:
            return HaliteShip.from_ship(ship)

    def shipyard_at_pos(self, pos: Point) -> Optional['HaliteShipyard']:
        cell = self.cells[pos]
        shipyard = cell.shipyard
        if shipyard is not None:
            return HaliteShipyard.from_shipyard(shipyard)

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
        return {id: HalitePlayer.from_player(p) for id, p in super().players.items()}\

    @property
    def player(self) -> 'HalitePlayer':
        return self.halite_players[self.current_player_id]


    @property
    def ships(self) -> Dict[ShipId, 'HaliteShip']:
        return {k: HaliteShip.from_ship(v) for k, v in super().ships.items()}

    @property
    def shipyards(self) -> Dict[ShipyardId, 'HaliteShipyard']:
        return {k: HaliteShipyard.from_shipyard(v) for k, v in super().shipyards.items()}

    @property
    def opponents(self) -> List['HalitePlayer']:
        return list(map(HalitePlayer.from_player, super().players))

    @property
    def opponent_ships(self) -> List['HaliteShip']:
        return np.flatten([p.ships for p in self.opponents])

    @property
    def opponent_shipyards(self) -> List['HaliteShipyard']:
        return np.flatten([p.shipyards for p in self.opponents])


from src.agent.entities.halite_ship import HaliteShip
from src.agent.entities.halite_shipyard import HaliteShipyard
from src.agent.entities.player import HalitePlayer

if __name__ == "__main__":
    a = np.zeros((10, 10, 10))

    a[:, 1, 1] = [i for i in range(10)]
