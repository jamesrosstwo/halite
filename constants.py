from pathlib import Path
import yaml


def load_settings():
    with open(Path("settings.yaml")) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


SETTINGS = load_settings()
ROOT_PATH = Path(__file__).resolve().parent