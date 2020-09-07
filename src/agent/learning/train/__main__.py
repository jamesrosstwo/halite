from kaggle_environments import make
from src.constants import ROOT_PATH, SETTINGS
from src.agent.learning.train.train_agent import HaliteTrainAgent
from src.agent.learning.train.memory import ReplayMemory
from typing import Dict, Any

ship_replay_memory = ReplayMemory(SETTINGS["learn"]["replay_memory_capacity"])
shipyard_replay_memory = ReplayMemory(SETTINGS["learn"]["replay_memory_capacity"])

steps_done = 0


def train_agent(observation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, str]:
    global steps_done
    agent = HaliteTrainAgent(observation, configuration, ship_replay_memory, shipyard_replay_memory, steps_done)
    steps_done += 1
    actions = agent.act()
    return actions


print("Making environment")
env = make("halite", debug=True)

print("Running agent")
env.run([train_agent, "random", "random", "random"])
ship_replay_memory.push_cache()
shipyard_replay_memory.push_cache()
