ml-ops
==============================

Simple ml-ops project

This is a simple Regression project to demontsrate the use of some Machine Learning Ops tools.

**DVC**

https://dvc.org/

DVC is built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code.

**Cookiecutter**

https://cookiecutter.readthedocs.io/en/1.7.2/

A command-line utility that creates projects from cookiecutters (project templates)

**Streamlit**

https://docs.streamlit.io/en/stable/index.html

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.


### Dataset 

The California Housing Prices dataset

https://www.kaggle.com/camnugent/california-housing-prices


Two simple models were evaluated ( A Linear Regression and a Random Forest). GridSearchCV was used to fine tune the Random Forest

The dataset  is tracked in DVC and is stored in a remote storage.

### Packages 

The Conda environment packages are exported in environment.yml

### App 

A simple app is created with Streamlit . To run it locally: 

**streamlit run app.py**


Project Organization (created with a Cookiecutter template)
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


