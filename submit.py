import json
import os
from pathlib import Path
from src.constants import ROOT_PATH, SETTINGS

# Nothing about this is good, but unfortunately this has to be compiled into one script for submission.
# There is a bug where if the files are in the wrong order in the directory, the imports will be broken and won't work.
# Need something like import_priority for each file to determine order to place in export script


if __name__ == "__main__":
    module_export_dir = ROOT_PATH / SETTINGS["submit"]["export_dir"]
    module_export_dir.mkdir(parents=True, exist_ok=True)

    module_export_path = module_export_dir / SETTINGS["submit"]["export_file_name"]
    module_export_path.touch(exist_ok=True)

    settings_str = "SETTINGS = " + str(json.dumps(SETTINGS))
    module_text = [settings_str]

    for root, dirs, files in os.walk(str(ROOT_PATH / (SETTINGS["agent"]["dir"])), topdown=False):
        for name in files:
            p = Path(root) / name
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

    os.system(submit_command)
