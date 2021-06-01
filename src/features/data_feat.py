import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import yaml
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def data_feat(config_path):
    config = read_params(config_path)
    vis_config = config["vis_data"]
    train_data_path = config["split_data"]["train_path"]

    prep_data_path = config["prepro_data"]["prepro_train"]
    pipeline_path = config["prepro_data"]["pipeline_path"]


    df = pd.read_csv(train_data_path, sep=",")
    df = df.drop('median_house_value',axis=1)
    sys.stderr.write(f'The input data frame  size is {df.shape}\n')



    housing_num = df.drop('ocean_proximity', axis=1)
    housing_cat = df[['ocean_proximity']]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])


    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])
    housing_prepared = full_pipeline.fit_transform(df)
    sys.stderr.write(f'The processed data frame  size is {housing_prepared.shape}\n')
    with open(pipeline_path, 'wb') as fd:
        pickle.dump(full_pipeline, fd)
    np.save(prep_data_path,housing_prepared)

    # sys.stderr.write(f'The processed data frame  size is {housing_prepared.shape}\n')



def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data_feat(config_path=parsed_args.config)