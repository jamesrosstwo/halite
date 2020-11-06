import base64
from pathlib import Path
import yaml
import torch

from submit import model_file_to_b64


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
SHIP_AGENT_B64_STRING = model_file_to_b64(_ship_agent_model_path)
SHIPYARD_AGENT_B64_STRING = model_file_to_b64(_shipyard_agent_model_path)

SHIP_AGENT_B64_STRING = str(SHIP_AGENT_B64_STRING)[2:-1]
SHIPYARD_AGENT_B64_STRING = str(SHIPYARD_AGENT_B64_STRING)[2:-1]
TORCH_DEVICE = torch.device("cuda")
