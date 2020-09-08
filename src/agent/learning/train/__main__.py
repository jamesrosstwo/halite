from kaggle_environments import make
from src.constants import SETTINGS, TORCH_DEVICE
from src.agent.learning.train.train_agent import HaliteTrainAgent
from src.agent.learning.ship_agent import HaliteShipAgent
from src.agent.learning.shipyard_agent import HaliteShipyardAgent
from src.agent.learning.train.memory import ReplayMemory
from src.agent.learning.train.optimizer import optimize_model

from typing import Dict, Any

ship_replay_memory = ReplayMemory(SETTINGS["learn"]["replay_memory_capacity"])
shipyard_replay_memory = ReplayMemory(SETTINGS["learn"]["replay_memory_capacity"])

steps_done = 0


def train_agent(observation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, str]:
    global steps_done
    agent = HaliteTrainAgent(observation, configuration, ship_replay_memory, shipyard_replay_memory, steps_done)
    steps_done += 1
    print(steps_done)
    actions = agent.act()
    return actions


print("Making environment")
env = make("halite", debug=True)

for i in range(SETTINGS["learn"]["num_train_episodes"]):
    print("Training step", i)
    print("Generating transition information")
    env.run([train_agent, "random", "random", "random"])
    print("Saving transition information")
    ship_replay_memory.push_cache()
    shipyard_replay_memory.push_cache()
    print("Ship agent optimization step")
    optimize_model(HaliteShipAgent().to(TORCH_DEVICE), ship_replay_memory)
    print("Shipyard agent optimization step")
    optimize_model(HaliteShipyardAgent().to(TORCH_DEVICE), shipyard_replay_memory)
