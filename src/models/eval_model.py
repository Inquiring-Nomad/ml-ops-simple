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
import pickle


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def eval(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    pipeline_path = config["prepro_data"]["pipeline_path"]
    model_dir = config["model_dir"]
    target = [config["base"]["target_col"]]
    df = pd.read_csv(test_data_path,sep=",")
    X_test = df.drop("median_house_value", axis=1)
    y_test = df["median_house_value"].copy()
    pipeline = pickle.load(open(pipeline_path, 'rb'))


    model_path = os.path.join(model_dir, "linear_regression.joblib")
    linear_reg_model =  joblib.load(model_path)
    X_test_preprocessed = pipeline.transform(X_test)
    final_predictions = linear_reg_model.predict(X_test_preprocessed)
    (rmse, mae, r2) = eval_metrics(y_test, final_predictions)
    ln_scores_file = config["reports"]["linreg_eval_scores"]
    with open(ln_scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    #Random Forrest
    rfr_model_path = os.path.join(model_dir, "rfr.joblib")
    rfr_model = joblib.load(rfr_model_path)
    X_test_preprocessed = pipeline.transform(X_test)
    final_rfr_predictions = rfr_model.predict(X_test_preprocessed)
    (rmse, mae, r2) = eval_metrics(y_test, final_rfr_predictions)
    rfr_scores_file = config["reports"]["rfr_eval_scores"]
    with open(rfr_scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)







def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    eval(config_path=parsed_args.config)