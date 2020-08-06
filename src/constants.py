from pathlib import Path
import yaml


def load_settings(root_path):
    with open(root_path / "settings.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


ROOT_PATH = Path(__file__).resolve().parent.parent
SETTINGS = load_settings(ROOT_PATH)
