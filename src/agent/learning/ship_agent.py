from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import SETTINGS

from kaggle_environments.envs.halite.helpers import ShipAction
from src.agent.entities.halite_ship import HaliteShip

SHIP_ACTION_MAP = {
    0: None,
    1: ShipAction.NORTH,
    2: ShipAction.EAST,
    3: ShipAction.SOUTH,
    4: ShipAction.WEST,
    5: ShipAction.CONVERT
}


def parse_input(ship: HaliteShip, board_input: np.ndarray):
    """
    Parses board state to NN input for this ship.

    Potential improvements:
        - Make more complex system for ship representation, right now it's just binary
        - Make the position representation a map of nothing but the current ship?
            Not sure if having ship position as x and y integers will be understood
    :param ship: HaliteShip to get things like position to generate input
    :param board_input: Board input as np.ndarray
    :return: 1d np.ndarray storing ship input
    """
    ship_input = board_input.copy()
    # Current player ship layer. Removing the current
    # ship allows the agent to more easily distinguish between
    # Itself and other ships
    ship_input[1, ship.position.x, ship.position.y] = 0

    # Encode player position
    final_ship_input = [ship.position.x, ship.position.y]
    final_ship_input = final_ship_input + list(ship_input.flatten())
    return torch.from_numpy(np.array(final_ship_input, dtype=np.float32))


class HaliteShipAgent(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(HaliteShipAgent, self).__init__()
        input_size = np.prod(SETTINGS["board"]["dims"]) + 2
        output_size = 6
        self.conv1 = nn.Linear(input_size, int(input_size * 1.2))
        self.conv2 = nn.Linear(int(input_size * 1.2), input_size // 80)
        self.output_layer = nn.Linear(input_size // 80, output_size)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.output_layer(y))
        return y

    def act(self, ship: HaliteShip, board_input: np.ndarray):
        ship_input = parse_input(ship, board_input)
        return self.forward(ship_input).argmax().item()
