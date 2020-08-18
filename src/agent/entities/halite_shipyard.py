from kaggle_environments.envs.halite.helpers import Shipyard, ShipyardId, Point, PlayerId, Board


class HaliteShipyard(Shipyard):
    def __init__(
            self,
            shipyard_id: ShipyardId,
            position: Point,
            player_id: PlayerId,
            board: 'Board',
            halite_board: 'HaliteBoard'
    ):
        super().__init__(shipyard_id, position, player_id, board)
        self._halite_board = halite_board

    @classmethod
    def from_shipyard(cls, shipyard_obj: Shipyard, halite_board: "HaliteBoard"):
        return cls(shipyard_obj.id, shipyard_obj.position, shipyard_obj.player_id, shipyard_obj._board, halite_board)

    def spawn(self):
        pass

    @property
    def player(self):
        from src.agent.entities.player import HalitePlayer
        return HalitePlayer.from_player(super().player)


from src.agent.board.board import HaliteBoard
