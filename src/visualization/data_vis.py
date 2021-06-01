import os
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
from pandas_profiling import ProfileReport



def visualise_data(config_path):
    config = read_params(config_path)
    vis_config = config["vis_data"]
    profiling_fname = vis_config["profiling_path"]

    train_data_path = config["split_data"]["train_path"]
    df = pd.read_csv(train_data_path, sep=",")
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file(profiling_fname)



def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    visualise_data(config_path=parsed_args.config)