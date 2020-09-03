import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import SETTINGS

from src.agent.entities.halite_shipyard import HaliteShipyard
from kaggle_environments.envs.halite.helpers import ShipyardAction

SHIPYARD_ACTION_MAP = {
    0: None,
    1: ShipyardAction.SPAWN
}


def parse_input(shipyard: HaliteShipyard, board_input: np.ndarray):
    """
    Parses board state to NN input for this shipyard.

    Potential improvements:
        - Make more complex system for shipyard representation, right now it's just binary
        - Make the position representation a map of nothing but the current shipyard?
            Not sure if having shipyard position as x and y integers will be understood
    :param shipyard: HaliteShipyard to get things like position to generate input
    :param board_input: Board input as np.ndarray
    :return: 1d np.ndarray storing shipyard input
    """
    shipyard_input = board_input.copy()
    # Current player shipyard layer. Removing the current
    # shipyard allows the agent to more easily distinguish between
    # Itself and other shipyards
    shipyard_input[5, shipyard.position.x, shipyard.position.y] = 0

    # Encode player position
    final_shipyard_input = [shipyard.position.x, shipyard.position.y]
    final_shipyard_input += shipyard_input.flatten()
    return final_shipyard_input


class HaliteShipyardAgent(nn.Module):
    """
    Agent to go from state to shipyard action
    """

    def __init__(self):
        super(HaliteShipyardAgent, self).__init__()
        input_size = np.prod(SETTINGS["board"]["dims"])
        output_size = 2
        batch_size = SETTINGS["learn"]["batch_size"]
        self.conv1 = nn.Conv1d(batch_size, input_size, 1)
        self.conv2 = nn.Conv1d(batch_size, int(input_size * 1.2), 1)
        self.conv3 = nn.Conv1d(batch_size, input_size // 80, 1)
        self.output_layer = nn.Conv1d(batch_size, output_size, 1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.output_layer(y))
        return y

    def act(self, shipyard: HaliteShipyard, board_input: np.ndarray):
        shipyard_input = parse_input(shipyard, board_input)
        return self.forward(shipyard_input)
