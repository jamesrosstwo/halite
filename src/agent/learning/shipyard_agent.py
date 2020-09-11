from abc import ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import SETTINGS, TORCH_DEVICE, SHIPYARD_AGENT_STATE_DICT

from src.agent.entities.halite_shipyard import HaliteShipyard
from kaggle_environments.envs.halite.helpers import ShipyardAction

SHIPYARD_ACTION_MAP = {
    0: None,
    1: ShipyardAction.SPAWN
}


def parse_shipyard_input(shipyard: HaliteShipyard, halite_board: "HaliteBoard", vision_dims=(21, 21)):
    """
    Parses board state to NN input for this shipyard.

    Potential improvements:
        - Make more complex system for shipyard representation, right now it's just binary
        - Make the position representation a map of nothing but the current shipyard?
            Not sure if having shipyard position as x and y integers will be understood
    :param shipyard: HaliteShipyard to get things like position to generate input
    :param halite_board: HaliteBoard to pull map data from
    :param vision_dims: How far in each direction the shipyard agent can see
    :return: 1d np.ndarray storing shipyard input
    """
    shipyard_input = halite_board.map.copy()
    board_center = (10, 10)
    center_shift = (shipyard.position.x - board_center[0], shipyard.position.y - board_center[1])
    centered_shipyard_input = np.roll(shipyard_input, center_shift, (1, 2))
    start_y = board_center[0] - vision_dims[0] // 2
    start_x = board_center[1] - vision_dims[1] // 2
    centered_shipyard_input = centered_shipyard_input[:, start_x:start_x + vision_dims[0],
                              start_y:start_y + vision_dims[1]]
    centered_shipyard_input = np.concatenate((centered_shipyard_input.flatten(), halite_board.additional_vals), axis=0)
    centered_shipyard_input = centered_shipyard_input.astype(np.float32)
    return torch.from_numpy(centered_shipyard_input).to(TORCH_DEVICE)


class HaliteShipyardAgent(nn.Module, metaclass=ABCMeta):
    """
    Agent to go from state to shipyard action
    """

    def __init__(self):
        super(HaliteShipyardAgent, self).__init__()
        self.input_channels = SETTINGS["board"]["dims"][0]
        self.vision_dims = (21, 21)
        output_size = 2
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3)
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(self.input_channels, self.input_channels * 2, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(77, 200)
        self.fc2 = nn.Linear(200, output_size)

    def forward(self, board_input):
        add_vals_size = SETTINGS["learn"]["num_additional_vals"]
        board_input_dims = SETTINGS["board"]["dims"]

        def feed_forward_input(fwd_input):
            additional_vals, x = fwd_input.split([add_vals_size, fwd_input.size(0) - add_vals_size])
            additional_vals = additional_vals.view(1, -1)
            x = x.view(1, self.input_channels, *self.vision_dims)

            in_size = x.size(0)
            x = self.conv1(x)
            x = F.relu(self.mp1(x))
            x = self.conv2(x)
            x = F.relu(self.mp2(x))
            x = F.relu(self.conv_drop(x))
            x = x.view(in_size, -1)

            x = torch.cat((x, additional_vals), dim=1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.log_softmax(x, dim=1)
            return x

        desired_input_sz = int(np.prod(board_input_dims) + add_vals_size)
        if board_input.size(0) == desired_input_sz:
            return feed_forward_input(board_input)
        out_tensors = []
        for state in board_input.split(desired_input_sz):
            out_tensors.append(torch.flatten(feed_forward_input(state)))
        return torch.stack(out_tensors)

    def act(self, shipyard: HaliteShipyard, board: "HaliteBoard"):
        shipyard_input = parse_shipyard_input(shipyard, board, self.vision_dims)
        return self.forward(shipyard_input).argmax().item()

    def copy(self):
        agent_copy = HaliteShipyardAgent().to(TORCH_DEVICE)
        agent_copy.load_state_dict(self.state_dict())
        return agent_copy

    def load_recent_model(self):
        self.load_state_dict(SHIPYARD_AGENT_STATE_DICT)


from src.agent.board.board import HaliteBoard
