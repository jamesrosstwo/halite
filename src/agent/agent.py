from kaggle_environments.envs.halite.helpers import Dict, Any
from src.agent.board.board import HaliteBoard

import torch


class HaliteAgent:
    def __init__(self, observation: Dict[str, Any], configuration: Dict[str, Any]):
        self.observation = observation
        self.configuration = configuration

        self.ship_states = {}
        self.halite_board = HaliteBoard(observation, configuration)
        self.player = self.halite_board.player
        self.ships = self.player.ships
        self.shipyards = self.player.shipyards

    def act(self) -> Dict[str, str]:
        from src.agent.learning.ship_agent import HaliteShipAgent, SHIP_ACTION_MAP
        from src.agent.learning.shipyard_agent import HaliteShipyardAgent, SHIPYARD_ACTION_MAP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ship_agent = HaliteShipAgent().to(device)
        shipyard_agent = HaliteShipyardAgent().to(device)
        for ship in self.ships:
            s_action = ship_agent.act(ship, self.halite_board.map)
            ship.next_action = SHIP_ACTION_MAP[s_action]
        for shipyard in self.shipyards:
            s_y_action = shipyard_agent.act(shipyard, self.halite_board.map)
            shipyard.next_action = SHIPYARD_ACTION_MAP[s_y_action]

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
