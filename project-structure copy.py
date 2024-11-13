import json
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import List

# creates the logging top directory if it does not exist
log_dir: str = "logs"
os.makedirs(log_dir, exist_ok=True)


# loads and configure logging
with open('logging-config.json', 'r') as f:
    config = json.load(f)
logging.config.dictConfig(config)

# gets this module logger
logger = logging.getLogger("project-template")


def create_structure(name: str) -> None:
    '''
    This functions creates this project directory structure
    '''

    # list of files to create:
    files: List[str] = [

        # for github actions
        ".github/workflows/.gitkeep",

        # source code structure
        f"{name}/__init__.py",
        f"{name}/src/__init__.py",
        f"{name}/src/components/__init__.py",
        f"{name}/src/config/__init__.py",
        f"{name}/src/utils/__init__.py",
        f"{name}/src/pipeline/__init__.py",
        f"{name}/src/cao/__init__.py",
        f"{name}/src/constants/__init__.py",
        f"{name}/src/architecture/__init__.py",

        # test structure
        "test/__init__.py",
        "test/components/__init__.py",
        "test/config/__init__.py",
        "test/utils/__init__.py",
        "test/pipeline/__init__.py",
        "test/cao/__init__.py",
        "test/constants/__init__.py",
        "test/architecture/__init__.py",

        # other
        "config/config.yaml",
        "laboratory/trials.ipynb",
        "params.yaml",
        "setup.py",
        "requirements.txt",
        "config/config.yaml",
    ]

    # Iterates over list of files
    for file in files:
        filepath: Path = Path(file)
        directory, filename = os.path.split(filepath)

        # first creates the directories
        if directory != "":
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory: {directory} created")

        # second creates the files
        if (not os.path.exists(path=filepath)):
            with open(file=filepath, mode="w"):
                logger.info(f"File {filepath} was created")
        else:
            logger.info(f"File {filepath} already exists. Skipping...")


if __name__ == ("__main__"):

    # Gets project name from args.
    project_name: str = sys.argv[1]

    # If project_name is not given, defaults to project home folder.
    if project_name is None:
        directory, project_name = os.path.split(Path(os.getcwd()))

    logger.info(f"Creating directory structure for project: {project_name}")

    # creates the directory
    create_structure(name=project_name)
