# Value players mlops

## Description


The objective of this project is to serve as a template for the use of MLOps. It will manage version control of both the code and the data, establish a workflow for training and retraining, as well as for obtaining metrics and versioning the model. CI/CD and unit tests will be applied. All of the above will be orchestrated and served via an API.


## Environment Setup

```shell
virtualenv venv
.\venv\Scripts\activate
pip install -r  requirements.txt
```

## Git and DVC Initialization

```
git init
dvc init
dvc remote add -d dvc-remote ./.dvc/tmp/dvc-storage
```

### Data Update

Each time there is a change in the dataset, execute the following commands:

```
dvc add ./data/data_fifa_players.json
dvc push
git add .
git commit -m "Add raw data"
```

## Pre-commit

Pre-commit will automatically run each time you try to make a commit in git. If pre-commit finds any issues in the files you are trying to commit, it will stop the commit and show you the issues so you can fix them.

```
# install pre-commit
pip install pre-commit
# config in local repositori
pre-commit install
# check after commit files
pre-commit run --all-files
```

## Tests

If you want to run test

```
pytest
```


## MLFlow Server

Start the MLFlow server

```
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Servidor Prefect

Start the Prefect server

```
prefect server start
```

## Documentation

Build the documentation

```
python -m pip install "mkdocstrings[python]"
pip install mkdocs-material
# Build mkdocs project
mkdocs new valueplayers_docs
mkdocs build
# run mkdocs server
python -m mkdocs serve
# visit http://127.0.0.1:8000/
```

## Docker

If necessary, create a container with a POSTGRES service to simulate model consumption monitoring.

```
cd monitoring
docker-compose up
```

Run the following file to simulate API usage

```
python monitoring/dummy_metrics_calculation.py
```
