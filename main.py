import copy
import json
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
from numpy.random import choice
from util.logging import logg_result, results_as_dict


with open("runtime.json") as file:
    variables = json.load(file)

def prediction_2017(temporal_validation, res):
    year_2017 = temporal_validation.copy()
    year_2017["forecast"] = res.prediction("validate", apply_inverse=True)
    year_2017["x"] = year_2017.apply(lambda x: x["year"] + x["month"] / 13, 1)
    year_2017.groupby("x").sum()[["value", "forecast"]].plot()
    plt.title("Validation Data: Insured Employment")
    plt.xlabel("Months")


def get_random_config(model_key):
    return {
        "rf": {
            "n_estimators": choice(range(5, 1001, 1)),
            "max_depth": None,
            "max_features": float(choice(range(1, 100, 1)) / 100),
            "min_samples_split": choice(range(2, 6, 1)),
            "min_samples_leaf": 1,
            "criterion": "mse",
            "n_jobs": -1
        },
        "gb": {
            "n_estimators": choice(range(50, 1001, 1)),
            "max_depth": choice(range(3, 60, 1)),
            "max_features": float(choice(range(1, 100, 1)) / 100),
            "min_samples_split": choice(range(2, 6, 1)),
            "min_samples_leaf": 1,
            "loss": choice(["ls", "lad", "huber"]),
            "learning_rate": choice(range(5, 31, 1)) / 100.0
        }
    }.get(model_key)


def get_percentage_error(res):
    return results_as_dict(res).get("model-performance").get("test").get("percentage-error")


def optimize_hyperparameters(model_key, loops, datasets, temporal_validation, speak=True):
    if speak:
        print("> Selected model: {}".format(model_key))
    best_config = get_random_config(model_key)
    best_model = regression_map[model_key].get("function")(best_config,
                                                           copy.deepcopy(datasets),
                                                           copy.deepcopy(temporal_validation))

    for i in range(loops):
        reg_conf = get_random_config(model_key)
        temp_model = regression_map[model_key].get("function")(
            reg_conf,
            copy.deepcopy(datasets),
            copy.deepcopy(temporal_validation))

        error = get_percentage_error(temp_model)
        if speak:
            print("\t* hyperparams optim loop ({}): {}".format(i+1, error))
        if get_percentage_error(best_model) > error:
            best_model = copy.deepcopy(temp_model)
            best_config = reg_conf
    return best_model, best_config


def main():

    """
    parser = OptionParser()
    parser.add_option("--model", type="string", default="rf", help="Select model.")
    parser.add_option("--plot", type="string", default="false", help="Plot.")
    kwargs, _ = parser.parse_args(args=None, values=None)
    # kwargs.model = "rf" #
    data, temporal_validation = get_data()
    datasets = DataSets(data,
                        encode_string=False,
                        one_hot_encode=False,
                        categ_to_num=True,
                        link="root_7",
                        transformations={'t-1': "root_7", 't-2': "root_7", 't-3': "root_7",
                                         't-6': "root_7", 't-7': "root_7"})
    reg_conf = get_regressor_conf(regression_map[kwargs.model].get("key"))
    res = regression_map[kwargs.model].get("function")(reg_conf, datasets, temporal_validation)
    logg_result(res, reg_conf, regression_map[kwargs.model].get("key"))
    if kwargs.plot.lower() == "true":
            prediction_2017(temporal_validation, res)
            res.error_distr("test")
            res.plot_estimations("test")
    """

    # TRAIN MODELS FOR THE NEXT 12 MONTHS
    model_periods = {}
    model_results = {}
    model_performance_validate = []
    production_model = {}
    for p in range(1, 13):
        data, temporal_validation, lags = get_data(min_lag=p - 1, save=False, read=True)
        test_cols = [c for c in data.columns if ("t-" in c) or ("value" in c) or ("year" in c) or ("month" in c)]
        datasets = DataSets(data,  # [test_cols],
                            encode_string=False,
                            one_hot_encode=False,
                            categ_to_num=True,
                            link=variables.get("link"),
                            transformations=variables.get("transformations"))
        model_key = variables.get("model")
        print("------------ Delta: " + str(p))
        res, reg_conf = optimize_hyperparameters(model_key, 20, datasets, temporal_validation)  # [test_cols])
        model_periods[p] = copy.deepcopy(res)
        model_results[p] = {
            "config": reg_conf,
            "general": results_as_dict(model_periods[p])
        }
        model_performance_validate.append(
            model_results[p].get("general").get("model-performance").get("validate").get("percentage-error"))
        del data
        del datasets
        del res

    model_periods = {}
    model_results = {}
    model_performance_validate = []
    production_model = {}
    with open("model_results.json") as file:
        model_results2 = json.load(file)
    for p in range(1, 13):
        data, temporal_validation, lags = get_data(min_lag=p - 1, save=False, read=True)
        comp_data = pd.concat([data, temporal_validation])

        dts = DataSets(comp_data,
                               encode_string=False,
                               one_hot_encode=False,
                               categ_to_num=True,
                               link=variables.get("link"),
                               transformations=variables.get("transformations"),
                               shuffle=True)
        reg_conf = {}# model_results2[str(p)].get("config")
        for k in model_results2[str(p)].get("config"):
            try:
                reg_conf[k] = int(model_results2[str(p)].get("config")[k]) if k != "max_features" else float(model_results2[str(p)].get("config")[k])
            except:
                reg_conf[k] = model_results2[str(p)].get("config")[k]
        pres = regression_map[variables.get("model")].get("function")(reg_conf, dts, temporal_validation)
        model_periods[p] = copy.deepcopy(pres)
        model_results[p] = {
            "config": reg_conf,
            "general": results_as_dict(model_periods[p])
        }
        del data
        del dts
        del pres


    def get_values_for(category, variable, model_results):
        core = []
        for p in range(1, 13):
            temp = model_results[p].get("general")["validation-data-2017"].get(category)
            row = {}
            for k in temp:
                if "Menor" in k:  # ignore menores
                    continue
                row[k] = temp[k].get(variable)
            core.append(row)
        return pd.DataFrame(core)


    # 12 MODELS PERFORMANCE
    model_performance_test = []
    model_performance_train = []
    for i in range(1, 13):
        model_performance_test.append(
            model_results[i].get("general").get("model-performance").get("test").get("percentage-error"))
        model_performance_train.append(
            model_results[i].get("general").get("model-performance").get("train").get("percentage-error"))

    pd.DataFrame({
        "test": model_performance_test,
        "train": model_performance_train,
        "locked-box": model_performance_validate
    }, index=range(1, 13)).plot.bar()
    # plt.plot(range(1, 13), model_performance_validate)
    plt.title("Model's percentage error")
    plt.xlabel("Models (n-periods ahead)")
    plt.ylabel("Total percentage error")
    plt.show()

    error_type = "percentage_error"
    df = get_values_for("age_range", error_type, model_results)
    df.plot(figsize=(10, 7)).legend(bbox_to_anchor=(0.75, 0.5))
    plt.title("Age-Range total percentage error in validation holdout")
    plt.ylabel("Total percentage error")
    plt.xlabel("Model (delta prediction)")
    plt.show()
    df = get_values_for("gender", error_type, model_results)
    df.plot(figsize=(10, 7)).legend(bbox_to_anchor=(0.75, 0.80))
    plt.title("Gender's total percentage error in validation holdout")
    plt.ylabel("Total percentage error")
    plt.xlabel("Model (delta prediction)")
    plt.show()
    df = get_values_for("economic_division", error_type, model_results)
    df.plot(figsize=(10, 7)).legend(bbox_to_anchor=(0.68, 0.60))
    plt.title("Economic Division total percentage error in validation holdout")
    plt.ylabel("Total percentage error")
    plt.xlabel("Model (delta prediction)")
    plt.show()

    class GeneralJsonEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.int64):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open("model_results.json", "w") as file:
        file.write(json.dumps(model_results, cls=GeneralJsonEncoder))

    def create_prediction_df(year, month, delta):
        cat_cols = ["economic_division", "age_range", "gender"]
        a, b, c = get_data(min_lag=0, save=False, read=True)
        complete_data = pd.concat([a, b])
        complete_data["time"] = complete_data.year + (complete_data.month-1) / 12
        complete_data = complete_data[[c for c in complete_data.columns if "t-" not in c]]
        req_lags = model_results[delta].get("general")["model-desc"].get("lags")

        def add_lags(sub_df, lags):
            original_index = sub_df.index
            response = sub_df.reset_index(drop=True)[["value"]]
            for lag in lags:
                temp = response[["value"]].iloc[:-lag]
                temp.index = temp.index + lag
                response["t-{}".format(lag)] = temp
            response.index = original_index
            del response["value"]
            return pd.concat([sub_df, response], axis=1)

        def add_lags_recursive(df, cols, lags, result_df=pd.DataFrame([])):
            unique_vals = df[cols[0]].unique()
            # sub_df = df.query("{} == '{}'".format("economic_division", "Servicios")).query("{} == '{}'".format("gender", "Mujeres")).query("{} == '{}'".format("age_range", "De 25 a 29 años."))
            for val in unique_vals:
                sub_df = df.query("{} == '{}'".format(cols[0], val))
                if len(cols) == 1:
                    max_time = sub_df.time.max()
                    if year + (month-1) / 12 > max_time:
                        for i in range(int(12 * (year + (month-1) / 12 - max_time))):
                            last = sub_df.iloc[-1]
                            row = {}
                            row["year"] = last.year + 1 if last.month == 12 else last.year
                            row["month"] = 1 if last.month == 12 else last.month + 1
                            row["time"] = row["year"] + (row["month"] - 1) / 12
                            row["value"] = float("nan")
                            for c in cat_cols:
                                row[c] = last[c]
                            sub_df = sub_df.append(pd.DataFrame([row], index=[sub_df.index.max() + 1]))

                    #result_df = pd.concat([result_df, add_lags(sub_df, lags)], axis=0)
                #else:
                    #result_df = add_lags_recursive(sub_df, cols[1:], lags, result_df=result_df)

                result_df = pd.concat([result_df, add_lags(sub_df, lags)], axis=0) \
                    if len(cols) == 1 else add_lags_recursive(sub_df, cols[1:], lags, result_df=result_df)
            return result_df

        dff = add_lags_recursive(complete_data.copy(), cat_cols, [int(a.replace("t-", "")) for a in req_lags])
        dff = dff.query("year == {} & month == {}".format(year, month))
        dff["value"] = 0
        return dff.dropna()








if __name__ == "__main__":
    main()
