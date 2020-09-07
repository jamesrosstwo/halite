from kaggle_environments import make
from src.constants import ROOT_PATH, SETTINGS

from typing import Dict, Any
from src.agent.agent import HaliteAgent


def train_agent(observation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, str]:
    agent = HaliteAgent(observation, configuration)
    actions = agent.act()
    return actions


print("Making environment")
env = make("halite", debug=True)

print("Running agent")
env.run([train_agent] * 4)

print("Saving replay")
render_options = SETTINGS["replay"]
game_path = ROOT_PATH / "games"
game_path.mkdir(parents=False, exist_ok=True)
game_output_path = game_path / "output.html"
f = open(game_output_path, "w")
f.write(env.render(**render_options))