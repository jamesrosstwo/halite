import base64
import json
import os
from pathlib import Path

from src.constants import *


# Nothing about this is good, but unfortunately this has to be compiled into one script for submission.
# There is a bug where if the files are in the wrong order in the directory, the imports will be broken and won't work.
# Need something like import_priority for each file to determine order to place in export script

def model_file_to_b64(state_dict_path):
    with open(state_dict_path, 'rb') as f:
        return base64.b64encode(f.read())


if __name__ == "__main__":
    module_export_dir = ROOT_PATH / SETTINGS["submit"]["export_dir"]
    module_export_dir.mkdir(parents=True, exist_ok=True)

    module_export_path = module_export_dir / SETTINGS["submit"]["export_file_name"]
    module_export_path.touch(exist_ok=True)

    SETTINGS["mode"] = "submit"
    settings_str = "SETTINGS = " + str(json.dumps(SETTINGS))
    torch_import_str = "import torch"
    torch_device_str = "TORCH_DEVICE = torch.device('cpu')"
    ship_agent_dict_str = "SHIP_AGENT_STATE_DICT = None"
    shipyard_agent_dict_str = "SHIPYARD_AGENT_STATE_DICT = None"

    _model_path = ROOT_PATH / SETTINGS["learn"]["models"]["save_dir"]
    _ship_agent_model_path = _model_path / SETTINGS["learn"]["models"]["ship_agent_file"]
    _shipyard_agent_model_path = _model_path / SETTINGS["learn"]["models"]["shipyard_agent_file"]
    SHIP_AGENT_B64_STRING = model_file_to_b64(_ship_agent_model_path)
    SHIPYARD_AGENT_B64_STRING = model_file_to_b64(_shipyard_agent_model_path)

    ship_agent_str = "SHIP_AGENT_B64_STRING = " + str(SHIP_AGENT_B64_STRING)
    shipyard_agent_str = "SHIPYARD_AGENT_B64_STRING = " + str(SHIPYARD_AGENT_B64_STRING)
    module_text = [settings_str, ship_agent_str, shipyard_agent_str, torch_import_str, torch_device_str, ship_agent_dict_str,
                   shipyard_agent_dict_str]

    path_blacklist = [ROOT_PATH / x for x in SETTINGS["submit"]["build_blacklist"]]

    for root, dirs, files in os.walk(str(ROOT_PATH / (SETTINGS["agent"]["dir"])), topdown=False):
        for name in files:
            p = Path(root) / name
            if any([x in p.parents for x in path_blacklist]):
                continue
            if p.is_file():
                if name == SETTINGS["agent"]["submission_file_name"]:
                    continue
                if name.split(".")[-1] == "py":
                    with open(p, 'r') as py_file:
                        module_text.append(py_file.read())

    with open(ROOT_PATH / SETTINGS["agent"]["dir"] / SETTINGS["agent"]["submission_file_name"], "r") as f:
        module_text.append(f.read())

    module_text_str = "\n".join(module_text)

    trim_module_text = []
    for line in module_text_str.split("\n"):
        if "src." not in line or "import" not in line:
            trim_module_text.append(line)

    module_text_str = "\n".join(trim_module_text)

    with open(str(module_export_path), "w") as module_export_file:
        module_export_file.write(module_text_str)
        module_export_file.close()

    submit_command = 'kaggle competitions submit -c halite -f ' + str(
        module_export_path) + ' -m "Deploy Script Submission"'

    # os.system(submit_command)
