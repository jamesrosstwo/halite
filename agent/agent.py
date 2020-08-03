# %%writefile submission.py

# Imports helper functions
from kaggle_environments.envs.halite.helpers import *
from agent.ship import HaliteShip



# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
ship_states = {}

# Returns the commands we send to our ships and shipyards
def agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player
    # print(board.step)
    # if board.step > 1:
    #     print(str(me.shipyards[0].cell.ship) == 'None')
    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.shipyards) > 0 and (len(me.ships) == 0 or (me.halite > 2000 and str(me.shipyards[0].cell.ship) == 'None')):
        me.shipyards[0].next_action = ShipyardAction.SPAWN

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT
    
    for ship in me.ships:
        if ship.next_action == None:
            
            ### Part 1: Set the ship's state 
            if ship.halite < 200: # If cargo is too low, collect halite
                ship_states[ship.id] = "COLLECT"
            if ship.halite > 500: # If cargo gets very big, deposit halite
                ship_states[ship.id] = "DEPOSIT"
                
            ### Part 2: Use the ship's state to select an action
            if ship_states[ship.id] == "COLLECT":
                # If halite at current location running low, 
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    neighbors = [ship.cell.north.halite, ship.cell.east.halite,
                                 ship.cell.south.halite, ship.cell.west.halite]
                    best = max(range(len(neighbors)), key=neighbors.__getitem__)
                    ship.next_action = directions[best]
            if ship_states[ship.id] == "DEPOSIT":
                # Move towards shipyard to deposit cargo
                direction = getDirTo(ship.position, me.shipyards[0].position, size)
                if direction: ship.next_action = direction
                
    return me.next_actions