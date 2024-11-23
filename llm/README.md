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

In order to create the structure the project must be initialized with poetry.

```bash
   poetry new llm
   ```

Then execute:
```bash
   python project-structure.py crawl_llm
   ```

The below structure is created:

├── README.md                 # Project documentation and overview </br>
├── llm/ </br>
│   ├── __init__.py           # Allows the directory to be treated as a package with poetry </br>
│   ├── llm/ </br>
│   |   ├── __init__.py </br>
│   |   ├── architecture/        # Folder containing the model architecture </br>
│   |   ├── cao/                 # Folder components configuration access objects </br>
│   |   ├── components/          # Folder containing components definitions </br>
│   |   ├── config/              # Folder containing the model configurations </br>
│   |   ├── constants/           # Folder containing general constants </br>
│   |   ├── exceptions/          # Folder containing custom exceptions </br>
│   |   ├── pipelines/           # Folder containing the pipelines </br>
│   |   |   ├── data_ingestion/  # Folder containing the data_ingestion pipeline </br>
│   ├── logs/                    # Folder containing all logs </br>
│   ├── models/                  # Folder containing trained models </br>
│   ├── notebooks/               # Folder containing notebooks for experimentation </br>
│   ├── reports/                 # Folder containing reports of this project </br>
│   ├── tests                    # Test folder </br>
│   |   ├── __init__.py </br>
│   |   ├── architecture/        # Folder containing the model architecture </br>
│   |   ├── cao/                 # Folder components configuration access objects </br>
│   |   ├── components/          # Folder containing components definitions </br>
│   |   ├── config/              # Folder containing the model configurations </br>
│   |   ├── constants/           # Folder containing general constants </br>
│   |   ├── exceptions/          # Folder containing custom exceptions </br>
│   |   ├── pipelines/           # Folder containing the pipelines </br>
│   |   |   ├── data_ingestion/  # Folder containing the data_ingestion pipeline </br>
|   ├── utils/                   # Helper functions for general utilities </br>
├── yamls/                       # Folder containing configuration yaml files </br>
├── .gitignore </br>
└── LICENSE                      # Project's LICENSE </br>

