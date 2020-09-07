from abc import ABCMeta
from typing import Any

from kaggle_environments.envs.halite.helpers import Dict
from src.agent.agent import HaliteAgent
from src.constants import TORCH_DEVICE
from src.agent.learning.ship_agent import HaliteShipAgent, parse_ship_input, SHIP_ACTION_MAP
from src.agent.learning.shipyard_agent import HaliteShipyardAgent, parse_shipyard_input, SHIPYARD_ACTION_MAP
from src.agent.learning.train.evaluator import evaluate_board


class HaliteTrainAgent(HaliteAgent):
    def __init__(self, observation: Dict[str, Any], configuration: Dict[str, Any], ship_memory, shipyard_memory):
        super().__init__(observation, configuration)
        self.ship_memory = ship_memory
        self.shipyard_memory = shipyard_memory

    def act(self) -> Dict[str, str]:
        ship_agent = HaliteTrainShipAgent(self.ship_memory).to(TORCH_DEVICE)
        shipyard_agent = HaliteTrainShipyardAgent(self.shipyard_memory).to(TORCH_DEVICE)
        for ship in self.ships:
            s_action = ship_agent.act(ship, self.halite_board)
            ship.next_action = SHIP_ACTION_MAP[s_action]
        for shipyard in self.shipyards:
            s_y_action = shipyard_agent.act(shipyard, self.halite_board)
            shipyard.next_action = SHIPYARD_ACTION_MAP[s_y_action]

        return self.get_next_actions()


class HaliteTrainShipAgent(HaliteShipAgent, metaclass=ABCMeta):
    def __init__(self, memory):
        super(HaliteTrainShipAgent, self).__init__()
        self.memory = memory

    def act(self, ship: "HaliteShip", board: "HaliteBoard"):
        self.halite_board = board
        ship_input = parse_ship_input(ship, board.map)
        action = self.forward(ship_input).argmax().item()
        self.memory.cache_state(ship.id, board.step, ship_input, action, evaluate_board(board))
        return action


class HaliteTrainShipyardAgent(HaliteShipyardAgent, metaclass=ABCMeta):
    def __init__(self, memory):
        super(HaliteTrainShipyardAgent, self).__init__()
        self.memory = memory

    def act(self, shipyard: "HaliteShipyard", board: "HaliteBoard"):
        self.halite_board = board
        shipyard_input = parse_shipyard_input(shipyard, board.map)
        action = self.forward(shipyard_input).argmax().item()
        self.memory.cache_state(shipyard.id, board.step, shipyard_input, action, evaluate_board(board))
        return action


from src.agent.entities.halite_ship import HaliteShip
from src.agent.entities.halite_shipyard import HaliteShipyard
from src.agent.board.board import HaliteBoard
