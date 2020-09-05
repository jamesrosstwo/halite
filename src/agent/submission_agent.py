from typing import Dict, Any
from src.agent.agent import HaliteAgent


def halite_agent(observation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, str]:
    agent = HaliteAgent(observation, configuration)
    actions = agent.act()
    return actions
