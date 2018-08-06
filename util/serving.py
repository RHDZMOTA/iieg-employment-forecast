import copy
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from conf import settings
from optparse import OptionParser
from util.logging import logg_result
from util.data_preparation import get_data, DataSets
from util.ml_models import regression_map
from util.regression import get_regressor_conf
from numpy.random import choice
from util.logging import logg_result, results_as_dict


class GeneralJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

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


def create_data():
    try:
        get_data(min_lag=0, save=True, read=False)
        return True
    except Exception as e:
        print(e)
        return False

def train_precedure():
    try:
        model_periods = {}
        model_results = {}
        model_performance_validate = []
        production_model = {}

        model_key = variables.get("model")

        for p in range(1, 13):
            data, temporal_validation, lags = get_data(min_lag=p - 1, save=False, read=True)
            test_cols = [c for c in data.columns if ("t-" in c) or ("value" in c) or ("year" in c) or ("month" in c)]
            datasets = DataSets(data,  # [test_cols],
                                encode_string=False,
                                one_hot_encode=False,
                                categ_to_num=True,
                                link=variables.get("link"),
                                transformations=variables.get("transformations"))

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

        with open("model_results_training.json", "w") as file:
            file.write(json.dumps(model_results, cls=GeneralJsonEncoder))

        return True
    except Exception as e:
        print(e)
        return False


def production_model_procedure():
    try:
        model_periods = {}
        model_results = {}
        model_performance_validate = []
        production_model = {}
        with open("model_results_training.json") as file:
            model_results_training = json.load(file)
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
            reg_conf = {}  # model_results_training[str(p)].get("config")
            for k in model_results_training[str(p)].get("config"):
                try:
                    reg_conf[k] = int(
                        model_results_training[str(p)].get("config")[k]) if k != "max_features" else float(
                        model_results_training[str(p)].get("config")[k])
                except:
                    reg_conf[k] = model_results_training[str(p)].get("config")[k]
            pres = regression_map[variables.get("model")].get("function")(reg_conf, dts, temporal_validation)
            model_periods[p] = copy.deepcopy(pres)
            model_results[p] = {
                "config": reg_conf,
                "general": results_as_dict(model_periods[p]),
                "model": pickle.dumps(pres.model).hex(),
                "cols": list(dts.get_train(False).columns)
            }
            del data
            del dts
            del pres

        with open("production_model.json", "w") as file:
            file.write(json.dumps(model_results, cls=GeneralJsonEncoder))
        return True
    except Exception as e:
        print(e)
        return False



def prediction_precedure():
    try:
        with open("production_model.json") as file:
            production_model = json.load(file)

        def create_prediction_df(year, month, delta):
            cat_cols = ["economic_division", "age_range", "gender"]
            a, b, c = get_data(min_lag=0, save=False, read=True)
            complete_data = pd.concat([a, b])
            complete_data["time"] = complete_data.year + (complete_data.month-1) / 12
            complete_data = complete_data[[c for c in complete_data.columns if "t-" not in c]]
            req_lags = production_model[str(delta)].get("general")["model-desc"].get("lags")

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

        training_data, temporal_validation, lags = get_data()
        database_df = pd.concat([training_data, temporal_validation])
        year  = int(database_df.year.max())
        month = int(database_df.month.max())
        prediction = pd.DataFrame([])
        for p in range(1, 3):
            if month < 12:
                month += 1
            else:
                month = 1
                year += 1
            model_index = str(p)
            data = create_prediction_df(year, month, p)
            model = pickle.loads(bytes.fromhex(production_model[model_index]["model"]))
            cols = production_model[model_index]["cols"]
            survive = ["time","year", "month", "age_range", "economic_division", "gender", "value"]
            dts = DataSets(data,
                                encode_string=False,
                                one_hot_encode=False,
                                categ_to_num=True,
                                link=variables.get("link"),
                                transformations=variables.get("transformations"),
                                shuffle=True)
            res = model.predict(dts.external(data[cols])[cols])
            data["value"] = dts.inverse_function[dts.link](res)
            prediction = pd.concat([prediction, data[survive]])
            del data
            del dts
            del model

        prediction.to_csv("output/last_prediction.csv")
        return True
    except Exception as e:
        print(e)
        return False
