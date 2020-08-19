from typing import List

from kaggle_environments.envs.halite.helpers import Player, PlayerId, ShipyardId, ShipId, Board



class HalitePlayer(Player):

    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: List[ShipyardId], ship_ids: List[ShipId],
                 board: Board, halite_board: "HaliteBoard"):
        super().__init__(player_id, halite, shipyard_ids, ship_ids, board)

        self._halite_board = halite_board
        self._halite_ships = self._halite_board.ships
        self._halite_shipyards = self._halite_board.shipyards

    @property
    def ships(self):
        return self._halite_ships

    @property
    def shipyards(self):
        return self._halite_shipyards

    @property
    def board(self):
        return self._halite_board

    @classmethod
    def from_player(cls, player_obj: Player, halite_board: "HaliteBoard"):
        return cls(player_obj.id, player_obj.halite, player_obj.shipyard_ids, player_obj.ship_ids, player_obj._board, halite_board)


    from src.agent.board.board import HaliteBoard
