import base64
from pathlib import Path
import yaml
import torch


def load_settings(root_path):
    with open(root_path / "settings.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


def load_model(model_string):
    # Write to temp file for kaggle submission
    with open("model.dat", "wb") as f:
        f.write(base64.b64decode(model_string))
        f.close()
    return torch.load('model.dat')


ROOT_PATH = Path(__file__).resolve().parent.parent
SETTINGS = load_settings(ROOT_PATH)

_model_path = ROOT_PATH / SETTINGS["learn"]["models"]["save_dir"]
_ship_agent_model_path = _model_path / SETTINGS["learn"]["models"]["ship_agent_file"]
_shipyard_agent_model_path = _model_path / SETTINGS["learn"]["models"]["shipyard_agent_file"]
SHIP_AGENT_STATE_DICT = torch.load(_ship_agent_model_path)
SHIPYARD_AGENT_STATE_DICT = torch.load(_shipyard_agent_model_path)
SHIP_AGENT_B64_STRING = ""
SHIPYARD_AGENT_B64_STRING = ""
TORCH_DEVICE = torch.device("cuda")
