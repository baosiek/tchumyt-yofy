# Large Language Model (LLM) Development Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Development Process](#development-process)
6. [Training and Evaluation](#training-and-evaluation)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

This project aims to develop, train, and deploy a Large Language Model (LLM) from scratch. This involves creating a natural language processing (NLP) model capable of understanding and generating human-like text based on a diverse corpus of data. The project follows a comprehensive pipeline, covering model development, testing, training, evaluation, and production.

## Project Structure
├── README.md              # Project documentation and overview
├── crawl_llm/
│   ├── __init__.py        # Allows the directory to be treated as a package with poetry
│   ├── crawl_llm/
│   |   ├── __init__.py 
│   |   ├── architecture/  # Folder containing the model architecture
│   |   ├── cao/           # Folder components configuration access objects
│   |   ├── components/    # Folder containing components definitions
│   |   ├── config/        # Folder containing the model configurations
│   |   ├── constants/     # Folder containing general constants
│   |   ├── pipelines/     # Folder containing the pipelines
│   ├── logs/              # Folder containing all logs
│   ├── models/            # Folder containing trained models
│   ├── notebooks/         # Folder containing notebooks for experimentation
│   ├── reports/           # Folder containing reports of this project
│   ├── tests              # Test folder
│   |   ├── __init__.py 
│   |   ├── architecture/  # Folder containing the model architecture
│   |   ├── cao/           # Folder components configuration access objects
│   |   ├── components/    # Folder containing components definitions
│   |   ├── config/        # Folder containing the model configurations
│   |   ├── constants/     # Folder containing general constants
│   |   └── pipelines/     # Folder containing the pipelines
|   ├── utils/             # Helper functions for general utilities   
├── yamls/                 # Folder containing configuration yaml files
├── .gitignore
└── LICENSE                # Project's LICENSE

