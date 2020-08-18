import torch.nn as nn
import torch.nn.functional as F

from src.agent.board.board import CURRENT_MAP

class HaliteShipAgent(nn.Module):
    def __init__(self, ship: "HaliteShip"):
        super(HaliteShipAgent, self).__init__()
        self.ship = ship
        self.conv1 = nn.Conv3d
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


from src.agent.entities.halite_ship import HaliteShip
