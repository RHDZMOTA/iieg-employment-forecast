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

def regressor_procedure(model, datasets):
    model.fit(datasets.get_train(), datasets.get_train(True))
    return RegressionResults(model, datasets)


class RegressionResults(object):

    train_label = settings.ModelConf.labels.train
    test_label = settings.ModelConf.labels.test
    validate_label = settings.ModelConf.labels.validate

    def __init__(self, model, datasets):
        self.model = model
        self.datasets = datasets

    def data(self, label):
        return self.datasets.get_train(False) if label == self.train_label else (
            self.datasets.get_test(False) if label == self.test_label else self.datasets.get_validate(False))

    def original_output(self, label):
        return self.datasets.get_train(True) if label == self.train_label else (
            self.datasets.get_test(True) if label == self.test_label else self.datasets.get_validate(True))

    def prediction(self, label):
        return self.model.predict(self.data(label))

    def error(self, label):
        return self.original_output(label) - self.prediction(label)

    def absolute_error(self, label):
        return np.abs(self.error(label))

    def square_error(self, label):
        return np.power(self.error(label), 2)

    def sme(self, label):
        return np.mean(self.square_error(label))

    def rsme(self, label):
        return np.sqrt(self.sme(label))

    def ame(self, label):
        return np.mean(self.absolute_error(label))

    def error_distr(self, label):
        errors = self.error(label)
        pd.DataFrame([e for e in errors if e < np.percentile(errors, 90)]).plot(kind="kde")
        plt.title("{} : Error Distribution".format(label))
        plt.show()

    def plot_estimations(self, label):
        real_list = self.original_output(label)
        estimate_list = self.prediction(label)
        plt.plot(real_list, real_list, ".b", label="real-values")
        plt.plot(real_list, estimate_list, ".g", alpha=0.7, label="estimations")
        plt.title("{} : Estimations vs Predictions".format(label))
        plt.legend()
        plt.xlabel("s")
        plt.ylabel("s")
        plt.show()



