# Nombre del Proyecto

## Descripción

Breve descripción del proyecto.

## Requisitos

Lista de requisitos o dependencias necesarias para ejecutar el proyecto.

## Configuración del entorno

```shell
virtualenv venv
.\venv\Scripts\activate
pip install -r  requirements.txt
```

## Inicialización de Git y DVC

```
git init
dvc init
dvc remote add -d dvc-remote ./.dvc/tmp/dvc-storage
```

### Actualización de datos

Cada vez que se realiza un cambio en el dataset, ejecuta los siguientes comandos:

```
dvc add ./data/data_fifa_players.json
dvc push
git add .
git commit -m "Add raw data"
```

## Pre-commit

pre-commit se ejecutará automáticamente cada vez que intentes hacer un commit en git. Si pre-commit encuentra algún problema en los archivos que estás intentando commitear, detendrá el commit y te mostrará los problemas para que puedas corregirlos.

```
# install pre-commit
pip install pre-commit
# config in local repositori
pre-commit install
# check after commit files
pre-commit run --all-files
```

## Pruebas

Descripción de cómo ejecutar las pruebas.

## Servidor MLFlow

Iniciar server de mlflow

```
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Servidor Prefect

Iniciar server de prefect

```
prefect server start
```

## Documentación

Construimos la documentación

```
python -m pip install "mkdocstrings[python]"
pip install mkdocs-material
# Build mkdocs project
mkdocs new valueplayers_docs
mkdocs build
# run mkdocs server
python -m mkdocs serve
```

## Docker

En caso de ser necesario se crea un contenedor con un
servicio de POSTGRES para simular el monitoreo de consumo
del modelo
```
cd monitoring
docker-compose up
```

Ejecutamos el siguiente archivo para simular la el uso
de la api

```
python monitoring/dummy_metrics_calculation.py
```
