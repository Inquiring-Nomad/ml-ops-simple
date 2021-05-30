import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import argparse
import joblib
import json
import yaml


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def train(config_path):
    config = read_params(config_path)
    target = [config["base"]["target_col"]]
    train_data_path = config["split_data"]["train_path"]
    train_pre_data_path = config["prepro_data"]["prepro_train"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestReg"]["param_grid"]["n_estimators"]
    max_features = config["estimators"]["RandomForestReg"]["param_grid"]["max_features"]
    cross_val_cv = config["estimators"]["LinearRegression"]["cross_val_cv"]
    param_grid = [
        {'n_estimators': n_estimators, 'max_features': max_features}

    ]
    df = pd.read_csv(train_data_path, sep=",")
    train_y= df[target]
    # train_x = pd.read_csv(train_pre_data_path, sep=",")
    train_x = np.load(train_pre_data_path)

    lin_reg = LinearRegression()
    lin_scores = cross_val_score(lin_reg, train_x, train_y,
                                 scoring="neg_mean_squared_error", cv=cross_val_cv)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)



    scores_file = config["reports"]["linreg_cross_val_scores"]


    with open(scores_file, "w") as f:
        scores = {
            "rmse": lin_rmse_scores.mean(),

        }
        json.dump(scores, f, indent=4)





    os.makedirs(model_dir, exist_ok=True)
    lin_reg.fit(train_x,train_y)
    model_path = os.path.join(model_dir, "linear_regression.joblib")

    joblib.dump(lin_reg, model_path)






def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train(config_path=parsed_args.config)