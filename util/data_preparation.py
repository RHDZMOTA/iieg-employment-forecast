import numpy as np
import pandas as pd

from conf import settings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

MONTHS = {
        "Enero": 1,
        "Febrero": 2,
        "Marzo": 3,
        "Abril": 4,
        "Mayo": 5,
        "Junio": 6,
        "Julio": 7,
        "Agosto": 8,
        "Septiembre": 9,
        "Octubre": 10,
        "Noviembre": 11,
        "Diciembre": 12,
    }


class DataSets(object):

    def __init__(self, data, split_vals={"train": 0.7, "test": 0.2, "validate": 0.1}, encode_string=True,
                 one_hot_encode = True, predictive_var="value"):
        data[predictive_var] = data[predictive_var].values.astype(np.float)
        self.predictive_var = predictive_var
        self.one_hot_encode = one_hot_encode
        self.string_encoder = {}
        self.one_hot_encoder = {}
        self.data = data
        if encode_string:
            self._encode_string()
        self.n, self.m = data.shape
        self._train_reference = split_vals["train"]
        self._test_validate_reference = split_vals["train"] + split_vals["test"]
        self._split_data()

    def _split_data(self):
        self.train, self.test, self.validate = np.split(
            self.data.sample(frac=1),
            [int(self.n * self._train_reference), int(self.n * self._test_validate_reference)])

    def get_train(self, output_var = False):
        if output_var:
            return self.train[self.predictive_var].values
        return self.train[self.train.columns[self.train.columns != self.predictive_var]]

    def get_test(self, output_var = False):
        if output_var:
            return self.test[self.predictive_var].values
        return self.test[self.test.columns[self.test.columns != self.predictive_var]]

    def get_validate(self, output_var = False):
        if output_var:
            return self.validate[self.predictive_var].values
        return self.validate[self.validate.columns[self.validate.columns != self.predictive_var]]

    def _encode_string(self):
        sub_data = self.data[self.data.columns[self.data.columns != self.predictive_var]]
        string_cols = sub_data.dtypes.to_frame()[(sub_data.dtypes.to_frame() == 'object').values].index.values
        for col in string_cols:
            LE = LabelEncoder()
            LE.fit(sub_data[col])
            classes = [col + "_" + str(i) for i in range(len(LE.classes_))]
            self.string_encoder[col] = LE
            self.data[col] = LE.transform(self.data[col])
            if self.one_hot_encode:
                OHE = OneHotEncoder()
                OHE.fit(self.data[col].values.reshape(-1, 1))
                vals = OHE.fit_transform(self.data[col].values.reshape(-1, 1)).toarray()
                temp = pd.DataFrame(vals, columns=classes)
                self.data = pd.concat([self.data, temp], axis=1)
                del self.data[col]
                self.one_hot_encoder[col] = OHE


def has_month(string):
    for month in MONTHS:
        if month.lower() in string.lower():
            return True
    return False


def get_data():
    raw_df = pd.read_pickle(settings.DataFilesConf.FileNames.insured_employment_pickle)
    value_cols = [col for col in raw_df.columns if has_month(col)]
    id_cols = [col for col in raw_df.columns if not has_month(col)]
    df = pd.melt(raw_df, id_vars=id_cols, value_vars=value_cols)
    df["year"], df["month"] = df.variable.str.split("_").str
    df["month"] = df["month"].replace(MONTHS)
    df["year"] = df["year"].values.astype(np.float)
    del df["variable"]
    return df

