from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import SETTINGS, TORCH_DEVICE

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


def parse_ship_input(ship: HaliteShip, board_input: np.ndarray, vision_dims=(21, 21)):
    """
    Parses board state to NN input for this ship.

    Potential improvements:
        - Make more complex system for ship representation, right now it's just binary
        - Make the position representation a map of nothing but the current ship?
            Not sure if having ship position as x and y integers will be understood
    :param vision_dims: How far in each direction the ship agent can see
    :param ship: HaliteShip to get things like position to generate input
    :param board_input: Board input as np.ndarray
    :return: 3d np.ndarray storing ship input
    """
    ship_input = board_input.copy()
    board_center = (10, 10)
    center_shift = (ship.position.x - board_center[0], ship.position.y - board_center[1])
    centered_ship_input = np.roll(ship_input, center_shift, (1, 2))
    start_y = board_center[0] - vision_dims[0] // 2
    start_x = board_center[1] - vision_dims[1] // 2
    centered_ship_input = centered_ship_input[:, start_x:start_x + vision_dims[0], start_y:start_y + vision_dims[1]]

    final_ship_input = np.expand_dims(centered_ship_input, 0)

    return torch.from_numpy(final_ship_input).to(TORCH_DEVICE)


class HaliteShipAgent(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(HaliteShipAgent, self).__init__()
        input_channels = SETTINGS["board"]["dims"][0]
        output_size = 6
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(77, 200)
        self.fc2 = nn.Linear(200, output_size)
        self.halite_board: "HaliteBoard" = None

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(self.mp1(x))
        x = self.conv2(x)
        x = F.relu(self.mp2(x))
        x = F.relu(self.conv_drop(x))
        x = x.view(in_size, -1)

        additional_vals = self.halite_board.get_additional_board_vals_tensor()
        x = torch.cat((x, additional_vals), dim=1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def act(self, ship: HaliteShip, board: "HaliteBoard"):
        self.halite_board = board
        ship_input = parse_ship_input(ship, board.map)
        return self.forward(ship_input).argmax().item()


from src.agent.board.board import HaliteBoard
