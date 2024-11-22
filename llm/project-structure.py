import os
import sys
import yaml

from pathlib import Path
from typing import Any, Dict, List


def create_directory(directory_name: str) -> None:
    '''

    This function creates the directory if is does not exist.

    input:
        directory_name: The name of the directory
        project_name: The project name from this script's list of arguments

    '''
    # Check if the directory exists. Creates it if it doesn't.
    if not Path(directory_name).exists():
        Path(directory_name).mkdir(parents=True, exist_ok=True)
        print(f"Directory {directory_name} was created.")
    else:
        print(f"Directory {directory_name} already exists.")

    # Checks if file __init__.py exists. Creates it if it doesn't
    if not Path(f"{directory_name}/__init__.py").exists():
        Path(f"{directory_name}/__init__.py").touch(exist_ok=True)
        print(f"File {f"{directory_name}/__init__.py"} was created.")
    else:
        print(f"File {f"{directory_name}/__init__.py"} already exists.")


def main(root_directory: str, project_name: str) -> None:
    '''

    This function reads yaml file and builds the project structure

    input:
        root_directory: The fully qualified path of the root directory
        relatively to the project directory.

    '''

    # reads the project yaml configuration file
    configuration_file: str = f"{root_directory}/project-structure.yaml"
    with open(configuration_file, 'r') as file:
        config: Dict[str, Any] = yaml.safe_load(file)

        # Gets and creates source directories
        src_dirs: List[str] = config['structure']['src']
        for dir in src_dirs:
            create_directory(f"{root_directory}/{project_name}/{dir}")

        # Gets and creates test directories
        src_dirs: List[str] = config['structure']['src']
        for dir in src_dirs:
            create_directory(f"{root_directory}/tests/{dir}")

        # Gets and creates others directories
        src_dirs: List[str] = config['structure']['others']
        for dir in src_dirs:
            create_directory(f"{root_directory}/{dir}")


if __name__ == ("__main__"):

    # Gets project name from args.
    project_name: str = sys.argv[1]

    # Gets current directory
    current_directory: str = os.getcwd()

    '''
    Checks if project was initialized accordingly with
    expected initial poetry project structure
    '''

    # Initializes root_directory
    root_directory: str = None
    # Check if the directory exists
    if os.path.exists(f"{current_directory}/{project_name}/{project_name}"):
        if os.path.exists(f"{current_directory}/{project_name}/tests"):
            print(f"Project {project_name} is correctly structured.")
            root_directory: str = f"{current_directory}/{project_name}"
            print(f"Root directory is {root_directory}")

    if root_directory is None:
        raise TypeError("Project is not correctly structured")

    # Executes this script
    main(root_directory=root_directory, project_name=project_name)
