import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conf import settings


def get_regressor_conf(regressor_key):
    setup_path = os.path.join(settings.PROJECT_DIR, "model_setup.json")
    with open(setup_path) as file:
        regressor_setup = json.load(file).get("regression")
    return regressor_setup.get(regressor_key)


def regressor_procedure(model, datasets, temporal_validation):
    model.fit(datasets.get_train(), datasets.get_train(True))
    return RegressionResults(model, datasets, temporal_validation)


class RegressionResults(object):

    train_label = settings.ModelConf.labels.train
    test_label = settings.ModelConf.labels.test
    validate_label = settings.ModelConf.labels.validate

    def __init__(self, model, datasets, temporal_validation):
        self.model = model
        self.datasets = datasets
        self.temporal_validation = temporal_validation
        self.train_data = self.datasets.get_train(False)
        self.test_data = self.datasets.get_test(False)
        self.validate_data = self.datasets.external(self.temporal_validation, False)

    def data(self, label):
        return self.train_data if label == self.train_label else (
            self.test_data if label == self.test_label else
            self.validate_data)

    def original_output(self, label, apply_inverse=False):
        return self.datasets.get_train(True, apply_inverse) if label == self.train_label else (
            self.datasets.get_test(True, apply_inverse) if label == self.test_label else
            self.datasets.external(self.temporal_validation, True, apply_inverse))

    def prediction(self, label, apply_inverse=False):
        prediction_values = self.model.predict(self.data(label))
        if apply_inverse:
            return self.datasets.inverse_function[self.datasets.link](prediction_values)
        return prediction_values

    def error(self, label, apply_inverse=False):
        return self.original_output(label, apply_inverse) - self.prediction(label, apply_inverse)

    def absolute_error(self, label, apply_inverse=False):
        return np.abs(self.error(label, apply_inverse))

    def square_error(self, label, apply_inverse=False):
        return np.power(self.error(label, apply_inverse), 2)

    def sme(self, label, apply_inverse=False):
        return np.mean(self.square_error(label, apply_inverse))

    def rsme(self, label, apply_inverse=False):
        return np.sqrt(self.sme(label, apply_inverse))

    def ame(self, label, apply_inverse=False):
        return np.mean(self.absolute_error(label, apply_inverse))

    def error_distr(self, label, apply_inverse=False):
        errors = self.error(label, apply_inverse)
        pd.DataFrame([e for e in errors if e < np.percentile(errors, 90)]).plot(kind="kde")
        plt.title("{} : Error Distribution".format(label))
        plt.show()

    def plot_estimations(self, label, apply_inverse=False):
        real_list = self.original_output(label, apply_inverse)
        estimate_list = self.prediction(label, apply_inverse)
        plt.plot(real_list, real_list, ".b", label="real-values")
        plt.plot(real_list, estimate_list, ".g", alpha=0.7, label="estimations")
        plt.title("{} : Estimations vs Predictions".format(label))
        plt.legend()
        plt.xlabel("s")
        plt.ylabel("s")
        plt.show()



