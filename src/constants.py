from pathlib import Path
import yaml
import torch


def load_settings(root_path):
    with open(root_path / "settings.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)



ROOT_PATH = Path(__file__).resolve().parent.parent
SETTINGS = load_settings(ROOT_PATH)

_model_path = ROOT_PATH / SETTINGS["learn"]["models"]["save_dir"]
_ship_agent_model_path = _model_path / SETTINGS["learn"]["models"]["ship_agent_file"]
_shipyard_agent_model_path = _model_path / SETTINGS["learn"]["models"]["shipyard_agent_file"]
SHIP_AGENT_STATE_DICT = torch.load(_ship_agent_model_path)
SHIPYARD_AGENT_STATE_DICT = torch.load(_shipyard_agent_model_path)
TORCH_DEVICE = torch.device("cuda")
