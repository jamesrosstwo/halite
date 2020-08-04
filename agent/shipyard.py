from kaggle_environments.envs.halite.helpers import Shipyard, ShipyardAction, ShipyardId, Point, PlayerId, Board


class HaliteShipyard(Shipyard):
    def __init__(self, shipyard_id: ShipyardId, position: Point, player_id: PlayerId, board: 'Board'):
        super().__init__(shipyard_id, position, player_id, board)

    @classmethod
    def from_shipyard(cls, shipyard_obj: Shipyard):
        return cls(shipyard_obj.id, shipyard_obj.position, shipyard_obj.player_id, shipyard_obj._board)

    def spawn(self):
        pass
