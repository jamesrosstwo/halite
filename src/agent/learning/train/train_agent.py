import math
import random

import numpy as np
import torch
from abc import ABCMeta
from typing import Any

from kaggle_environments.envs.halite.helpers import Dict
from src.agent.agent import HaliteAgent
from src.constants import TORCH_DEVICE, SETTINGS
from src.agent.learning.ship_agent import HaliteShipAgent, parse_ship_input, SHIP_ACTION_MAP
from src.agent.learning.shipyard_agent import HaliteShipyardAgent, parse_shipyard_input, SHIPYARD_ACTION_MAP
from src.agent.learning.train.evaluator import evaluate_board


class HaliteTrainAgent(HaliteAgent):
    def __init__(self, observation: Dict[str, Any], configuration: Dict[str, Any], ship_mem, shipyard_mem, n_steps):
        super().__init__(observation, configuration)
        self.ship_memory = ship_mem
        self.shipyard_memory = shipyard_mem
        self.step_no = n_steps

    def act(self) -> Dict[str, str]:
        ship_agent = HaliteTrainShipAgent(self.ship_memory).to(TORCH_DEVICE)
        shipyard_agent = HaliteTrainShipyardAgent(self.shipyard_memory).to(TORCH_DEVICE)

        e_end = SETTINGS["learn"]["eps_end"]
        e_start = SETTINGS["learn"]["eps_start"]
        e_decay = SETTINGS["learn"]["eps_decay"]
        eps_threshold = e_end + (e_start - e_end) * math.exp(-1. * self.step_no / e_decay)

        for ship in self.ships:
            sample = random.random()
            if sample > eps_threshold:
                s_action = ship_agent.act(ship, self.halite_board)
                ship.next_action = SHIP_ACTION_MAP[s_action]
            else:
                ship.next_action = SHIP_ACTION_MAP[random.randint(0, 5)]
        for shipyard in self.shipyards:
            sample = random.random()
            if sample > eps_threshold:
                s_y_action = shipyard_agent.act(shipyard, self.halite_board)
                shipyard.next_action = SHIPYARD_ACTION_MAP[s_y_action]
            else:
                shipyard.next_action = SHIPYARD_ACTION_MAP[random.randint(0, 1)]

        return self.get_next_actions()


class HaliteTrainShipAgent(HaliteShipAgent, metaclass=ABCMeta):
    def __init__(self, memory):
        super(HaliteTrainShipAgent, self).__init__()
        self.memory = memory

    def act(self, ship: "HaliteShip", board: "HaliteBoard"):
        ship_input = parse_ship_input(ship, board)
        forward_res = self.forward(ship_input)

        action = torch.tensor([forward_res.argmax().item()], device=TORCH_DEVICE)
        value = torch.tensor([evaluate_board(board)], device=TORCH_DEVICE)
        self.memory.cache_state(ship.id, board.step, ship_input, action, value)
        return action.argmax().item()


class HaliteTrainShipyardAgent(HaliteShipyardAgent, metaclass=ABCMeta):
    def __init__(self, memory):
        super(HaliteTrainShipyardAgent, self).__init__()
        self.memory = memory

    def act(self, shipyard: "HaliteShipyard", board: "HaliteBoard"):
        shipyard_input = parse_shipyard_input(shipyard, board)
        forward_res = self.forward(shipyard_input)

        action = torch.tensor([forward_res.argmax().item()], device=TORCH_DEVICE)
        value = torch.tensor([evaluate_board(board)], device=TORCH_DEVICE)
        self.memory.cache_state(shipyard.id, board.step, shipyard_input, action, value)
        return action.argmax().item()


from src.agent.entities.halite_ship import HaliteShip
from src.agent.entities.halite_shipyard import HaliteShipyard
from src.agent.board.board import HaliteBoard
