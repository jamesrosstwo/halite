from typing import List

from kaggle_environments.envs.halite.helpers import Player, PlayerId, ShipyardId, ShipId, Board


class HalitePlayer(Player):

    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: List[ShipyardId], ship_ids: List[ShipId],
                 board: Board, halite_board: "HaliteBoard"):
        super().__init__(player_id, halite, shipyard_ids, ship_ids, board)

        self._halite_board = halite_board
        self._halite_ships = self.get_halite_ships()
        self._halite_shipyards = self.get_halite_shipyards()

    def get_halite_ships(self):
        all_ships = self._halite_board.ships
        return [x for x in all_ships if x.player_id == self.id]

    def get_halite_shipyards(self):
        all_shipyards = self.board.shipyards
        return [x for x in all_shipyards if x.player_id == self.id]

    @property
    def ships(self) -> List["HaliteShip"]:
        return self._halite_ships

    @property
    def shipyards(self) -> List["HaliteShipyard"]:
        return self._halite_shipyards

    @property
    def board(self) -> "HaliteBoard":
        return self._halite_board

    @classmethod
    def from_player(cls, player_obj: Player, halite_board: "HaliteBoard"):
        return cls(player_obj.id, player_obj.halite, player_obj.shipyard_ids, player_obj.ship_ids, player_obj._board,
                   halite_board)

    from src.agent.board.board import HaliteBoard
    from src.agent.entities.halite_ship import HaliteShip
    from src.agent.entities.halite_shipyard import HaliteShipyard
