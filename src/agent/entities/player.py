from typing import List

from kaggle_environments.envs.halite.helpers import Player, PlayerId, ShipyardId, ShipId, Board

from src.agent.entities.halite_ship import HaliteShip
from src.agent.entities.halite_shipyard import HaliteShipyard


class HalitePlayer(Player):

    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: List[ShipyardId], ship_ids: List[ShipId],
                 board: Board):
        super().__init__(player_id, halite, shipyard_ids, ship_ids, board)

    @property
    def ships(self) -> List['HaliteShip']:
        return [HaliteShip.from_ship(x) for x in super().ships]

    @property
    def shipyards(self) -> List['HaliteShipyard']:
        return [HaliteShipyard.from_shipyard(x) for x in super().shipyards]

    @classmethod
    def from_player(cls, player_obj: Player):
        return cls(player_obj.id, player_obj.halite, player_obj.shipyard_ids, player_obj.ship_ids, player_obj._board)
