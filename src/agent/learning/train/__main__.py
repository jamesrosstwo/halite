from kaggle_environments import make
from src.constants import SETTINGS, TORCH_DEVICE, ROOT_PATH
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
    if steps_done % 50 == 0:
        print("#", end="")
    actions = agent.act()
    return actions


print("Making environment")
env = make("halite", debug=True)

ship_agent = HaliteShipAgent().to(TORCH_DEVICE)
shipyard_agent = HaliteShipyardAgent().to(TORCH_DEVICE)

# ship_agent.load_recent_model()
# shipyard_agent.load_recent_model()

for i in range(SETTINGS["learn"]["num_train_episodes"]):
    print("Training step", i)
    print("Generating transition information")
    env.run([train_agent, "random", "random", "random"])
    print()
    print("Saving transition information")
    ship_replay_memory.push_cache()
    shipyard_replay_memory.push_cache()
    print("Ship agent optimization step")
    model_path = ROOT_PATH / SETTINGS["learn"]["models"]["save_dir"]
    ship_agent_model_path = model_path / SETTINGS["learn"]["models"]["ship_agent_file"]
    shipyard_agent_model_path = model_path / SETTINGS["learn"]["models"]["shipyard_agent_file"]
    optimize_model(ship_agent, ship_replay_memory, ship_agent_model_path)
    print("Shipyard agent optimization step")
    optimize_model(shipyard_agent, shipyard_replay_memory, shipyard_agent_model_path)
