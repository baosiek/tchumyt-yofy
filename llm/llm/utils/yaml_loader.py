import yaml

from typing import Dict, Any


def get_yaml_file(pathname: str) -> Dict[str, Any]:
    with open(pathname, 'r') as file:
        return yaml.safe_load(file)
