import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
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


    #####Random Forrest#####
    param_grid = [

        {'n_estimators': n_estimators, 'max_features': max_features}
    ]

    forest_reg = RandomForestRegressor(random_state=random_state)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=cross_val_cv,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(train_x,train_y)

    rfr_best_score = grid_search.best_score_
    rfr_best_est = grid_search.best_estimator_

    rfr_rmse_scores = np.sqrt(-rfr_best_score)





    lin_scores_file = config["reports"]["linreg_cross_val_scores"]
    rfr_scores_file = config["reports"]["rfr_cross_val_scores"]


    with open(lin_scores_file, "w") as f:
        scores = {
            "rmse": lin_rmse_scores.mean(),

        }
        json.dump(scores, f, indent=4)
    with open(rfr_scores_file, "w") as f:
        scores = {
            "rmse": rfr_rmse_scores,

        }
        json.dump(scores, f, indent=4)





    os.makedirs(model_dir, exist_ok=True)
    lin_reg.fit(train_x,train_y)
    lin_model_path = os.path.join(model_dir, "linear_regression.joblib")
    rfr_model_path = os.path.join(model_dir, "rfr.joblib")

    joblib.dump(lin_reg, lin_model_path)
    joblib.dump(rfr_best_est, rfr_model_path)






def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train(config_path=parsed_args.config)