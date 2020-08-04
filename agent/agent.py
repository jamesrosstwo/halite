# %%writefile submission.py

# Imports helper functions
from kaggle_environments.envs.halite.helpers import *
from agent.ship import HaliteShip, HaliteShipState
from agent.shipyard import HaliteShipyard
from board.board import HaliteBoard


class HaliteAgent:
    def __init__(self, observation: Dict[str, Any], configuration: Dict[str, Any]):
        self.observation = observation
        self.configuration = configuration

        self.ship_states = {}
        self.halite_board = HaliteBoard(observation, configuration)
        self.player = self.halite_board.board_obj.current_player
        self.ships = [HaliteShip.from_ship(x) for x in self.player.ships]
        self.shipyards = [HaliteShipyard.from_shipyard(x) for x in self.player.shipyards]

    def act(self) -> Dict[str, str]:
        for ship in self.ships:
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


def halite_agent(observation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, str]:
    halite_agent = HaliteAgent(observation, configuration)
    actions = halite_agent.act()
    return actions