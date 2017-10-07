import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from conf import settings
from optparse import OptionParser
from util.logging import logg_result
from sklearn.neural_network import MLPRegressor
from util.data_preparation import get_data, DataSets
from util.regression import regressor_procedure, get_regressor_conf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

train_label = settings.ModelConf.labels.train
test_label = settings.ModelConf.labels.test
validate_label = settings.ModelConf.labels.validate


def random_forest(reg_conf, datasets):
    model = RandomForestRegressor(
        n_estimators=reg_conf["n-estimators"],
        n_jobs=reg_conf["n-jobs"]
    )
    res = regressor_procedure(model, datasets)
    return res


def mlp(reg_conf, datasets):
    model = MLPRegressor(
        hidden_layer_sizes=eval(reg_conf["hidden-layers"]),
        max_iter=reg_conf["max-iter"],
        activation=reg_conf["activation-function"]
    )
    res = regressor_procedure(model, datasets)
    return res


regression_map = {
    "rf": {
        "key": "random-forest",
        "function": random_forest
    },
    "mlp": {
        "key": "mlp",
        "function": mlp
    }
}


def main():
    parser = OptionParser()
    parser.add_option("--model", type="string", help="Select model.")
    parser.add_option("--plot", type="string", help="Select model.")
    kwargs, _ = parser.parse_args(args=None, values=None)
    data = get_data().query("value != 'N/D'").reset_index(drop=True)
    datasets = DataSets(data)
    reg_conf = get_regressor_conf(regression_map[kwargs.model].get("key"))
    res = regression_map[kwargs.model].get("function")(reg_conf, datasets)
    logg_result(res, reg_conf, regression_map[kwargs.model].get("key"))
    if kwargs.plot is not None:
        res.error_distr("test")
        res.plot_estimations("test")


if __name__ == "__main__":
    main()
