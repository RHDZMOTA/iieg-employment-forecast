import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conf import settings
from optparse import OptionParser
from util.logging import logg_result
from util.data_preparation import get_data, DataSets
from util.ml_models import regression_map
from util.regression import get_regressor_conf


def prediction_2017(temporal_validation, res):
    year_2017 = temporal_validation.copy()
    year_2017["forecast"] = res.prediction("validate", apply_inverse=True)
    year_2017.groupby("month").sum()[["value", "forecast"]].plot()
    plt.title("Validation Data: Insured Employment 2017")
    plt.xlabel("Months")


def main():
    parser = OptionParser()
    parser.add_option("--model", type="string", default="rf", help="Select model.")
    parser.add_option("--plot", type="string", default="false", help="Plot.")
    kwargs, _ = parser.parse_args(args=None, values=None)
    # kwargs.model = "rf" #
    data, temporal_validation = get_data()
    datasets = DataSets(data,
                        link="root_7",
                        transformations={'t-1': "root_7", 't-2': "root_7", 't-3': "root_7", 't-6': "root_7", 't-7': "root_7"})
    reg_conf = get_regressor_conf(regression_map[kwargs.model].get("key"))
    res = regression_map[kwargs.model].get("function")(reg_conf, datasets, temporal_validation)
    logg_result(res, reg_conf, regression_map[kwargs.model].get("key"))
    if kwargs.plot.lower() == "true":
            prediction_2017(temporal_validation, res)
            res.error_distr("test")
            res.plot_estimations("test")


if __name__ == "__main__":
    main()
