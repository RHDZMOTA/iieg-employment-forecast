from conf import settings
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import VotingClassifier
from util.regression import regressor_procedure
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from util.regression import regressor_procedure, get_regressor_conf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


train_label = settings.ModelConf.labels.train
test_label = settings.ModelConf.labels.test
validate_label = settings.ModelConf.labels.validate


def random_forest(reg_conf, datasets, temporal_validation):
    model = RandomForestRegressor(
        n_estimators=reg_conf["n_estimators"],
        max_depth=reg_conf["max_depth"],
        max_features=reg_conf["max_features"],
        min_samples_split=reg_conf["min_samples_split"],
        min_samples_leaf=reg_conf["min_samples_leaf"],
        criterion=reg_conf["criterion"],
        n_jobs=reg_conf["n_jobs"]
    )
    res = regressor_procedure(model, datasets, temporal_validation)
    return res


def gradient_boosting(reg_conf, datasets, temporal_validation):
    model = GradientBoostingRegressor(
        n_estimators=reg_conf["n_estimators"],
        max_depth=reg_conf["max_depth"],
        max_features=reg_conf["max_features"],
        min_samples_split=reg_conf["min_samples_split"],
        min_samples_leaf=reg_conf["min_samples_leaf"],
        loss=reg_conf["loss"],
        learning_rate=reg_conf["learning_rate"]
    )
    res = regressor_procedure(model, datasets, temporal_validation)
    return res

#def mlp(reg_conf, datasets, temporal_validation):
#    model = MLPRegressor(
#        hidden_layer_sizes=eval(reg_conf["hidden-layers"]),
#        max_iter=reg_conf["max-iter"],
#        activation=reg_conf["activation-function"]
#    )
#    res = regressor_procedure(model, datasets, temporal_validation)
#    return res


#def xgradient_boost_regression(reg_conf, datasets, temporal_validation):
#    model = XGBRegressor()
#    res = regressor_procedure(model, datasets, temporal_validation)
#    return res


regression_map = {
    "rf": {
        "key": "random-forest",
        "function": random_forest
    },
    "gb": {
        "key": "gradient-boosting",
        "function": gradient_boosting
    }
    #"mlp": {
    #    "key": "mlp",
    #    "function": mlp
    #},
    #"xgb": {
    #    "key": "xgboost",
    #    "function": xgradient_boost_regression
    #}
}


