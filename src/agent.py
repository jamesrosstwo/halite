from kaggle_environments.envs.halite.helpers import *
from src.board.board import HaliteBoard
from src.constants import SETTINGS
import torch

from src.entities.ship import HaliteShipState


def ModGrid(grid, startX, startY, radius, value):
    for i in range(startY-radius, startY+radius+1):
        for j in range(startX-radius, startX+radius+1):
            grid[(i+21) % 21][(j+21) % 21] += value

class HaliteAgent:
    def __init__(self, observation: Dict[str, Any], configuration: Dict[str, Any]):
        self.observation = observation
        self.configuration = configuration

        self.ship_states = {}
        self.halite_board = HaliteBoard(observation, configuration)
        self.player = self.halite_board.player
        self.ships = self.player.ships
        self.shipyards = self.player.shipyards

    # def recurse(self, x, y, step):


    def act(self) -> Dict[str, str]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.get_next_actions()

    def get_next_actions(self) -> Dict[str, str]:
        # zoneOfControl = [[self.halite_board.cells[(i,j)].halite for i in range(21)] for j in range(21)]
        # print(self.halite_board.cells[(0,0)])
        # for next in self.halite_board.ships.values():
            # print(ship[1])
            # zoneOfControl[ship.position[0]][ship.position[1]] = ship.player_id
            # ModGrid(zoneOfControl, next.position[0], next.position[1], 1, 50 if next.player_id == 0 else -50)
            # ModGrid(zoneOfControl, next.position[0], next.position[1], 0, -1000)
        # for next in self.halite_board.shipyards.values():
        #     ModGrid(zoneOfControl, next.position[0], next.position[1], 2, 50 if next.player_id == 0 else -50)
        #     ModGrid(zoneOfControl, next.position[0], next.position[1], 0, -1000 if next.player_id != 0 else 0)
        # for next in zoneOfControl:
        #     print(next)
            # If there are no ships, use first shipyard to spawn a ship.
        if len(self.shipyards) > 0 and (
                len(self.ships) == 0 or (self.player.halite > (2000+self.halite_board.step*10) and str(self.shipyards[0].cell.ship) == 'None')):
            self.shipyards[0].spawn()
        # If there are no shipyards, convert first ship into shipyard.
        # print(self.shipyards[0].position)
        # print(self.ships[0].position)
        if len(self.shipyards) == 0 and len(self.ships) > 0:
            self.ships[0].convert()
        for ship in self.ships:
            if ship.next_action is not None: continue
            if ship.halite < 200:  # If cargo is too low, collect halite
                ship.state = HaliteShipState.COLLECT
            if ship.halite > 500:  # If cargo gets very big, deposit halite
                ship.state = HaliteShipState.DEPOSIT

            if ship.state == HaliteShipState.COLLECT:
                # If halite at current location running low,
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    ship.move_to_max_adjacent_halite()
            if ship.state == HaliteShipState.DEPOSIT:
                # Deposit to the first shipyard
                if len(self.shipyards) == 0:
                    ship.next_action = None
                    continue
                direction = ship.get_dir_to(self.shipyards[0].position)
                if direction: ship.next_action = direction
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


