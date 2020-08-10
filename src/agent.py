from kaggle_environments.envs.halite.helpers import *
from src.board.board import HaliteBoard

import torch


class HaliteAgent:
    def __init__(self, observation: Dict[str, Any], configuration: Dict[str, Any]):
        self.observation = observation
        self.configuration = configuration

        self.ship_states = {}
        self.halite_board = HaliteBoard(observation, configuration)
        self.player = self.halite_board.player
        self.ships = self.player.ships
        self.shipyards = self.player.ships
        
    def act(self) -> Dict[str, str]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.get_next_actions()

    def get_next_actions(self) -> Dict[str, str]:
        ship_actions = {
            ship.id: ship.next_action.name
            for ship in self.ships
            if ship.next_action is not None
        }
        shipyard_actions = {
            shipyard.id: shipyard.next_action.name
            for shipyard in self.shipyards
            if shipyard.next_action is not None
        }
        return {**ship_actions, **shipyard_actions}

    def get_ship_states(self):
        return {x.id: x.state for x in self.ships}


