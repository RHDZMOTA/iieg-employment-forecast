from conf import settings
import pandas as pd
import numpy as np
import datetime
import os


def stringify_results(res, reg_conf, regression_key):
    res_string = """

    -------------------------------
    {datetime}

    SELECTED MODEL: {model}

    Link Function (y-transform): {link}
    Other Transformations (x-transform): 
{transf}

    PRAMETERS:
{params}

    TRAIN DATA
    > SME : {sme_train} ({sme_train_before})
    > RSME: {rsme_train} ({rsme_train_before})
    > AME : {ame_train} ({ame_train_before})

    TEST DATA
    > SME : {sme_test} ({sme_test_before})
    > RSME: {rsme_test} ({rsme_test_before})
    > AME : {ame_test} ({ame_test_before})

    TEMPORAL VALIDATION (2017)
    > SME : {sme_valid} ({sme_valid_before})
    > RSME: {rsme_valid} ({rsme_valid_before})
    > AME : {ame_valid} ({ame_valid_before})

    Response Variable Stats (insured employment) -- train data
    Stats:
    {stats}

    Temp. Validation RMSE / response_mean = {mean}
    Temp. Validation RMSE / response_median = {median}
    """
    # Response Variable Stats
    stats = pd.DataFrame(res.datasets.get_train(True, True), columns=["response-variable"]).describe()
    # Stringify Parameters
    params = ""
    for param in reg_conf:
        params += "\t> " + param + ": " + str(reg_conf[param]) + "\n"
    # Stringify x-transforms
    other_transf = ""
    tranf_functions = res.datasets.transformations
    for transf in tranf_functions:
        other_transf += "\t> " + transf + ": " + str(tranf_functions[transf]) + "\n"
    # Format Content
    now = datetime.datetime.now()
    content = res_string.format(
        datetime=now.strftime("%Y/%m/%d %H:%M:%S"),
        model=regression_key,
        link=res.datasets.link,
        transf=other_transf,
        params=params,
        sme_train=res.sme(settings.ModelConf.labels.train, apply_inverse=True),
        sme_train_before=res.sme(settings.ModelConf.labels.train, apply_inverse=False),
        rsme_train=res.rsme(settings.ModelConf.labels.train, apply_inverse=True),
        rsme_train_before=res.rsme(settings.ModelConf.labels.train, apply_inverse=False),
        ame_train=res.ame(settings.ModelConf.labels.train, apply_inverse=True),
        ame_train_before=res.ame(settings.ModelConf.labels.train, apply_inverse=False),
        sme_test=res.sme(settings.ModelConf.labels.test, apply_inverse=True),
        sme_test_before=res.sme(settings.ModelConf.labels.test, apply_inverse=False),
        rsme_test=res.rsme(settings.ModelConf.labels.test, apply_inverse=True),
        rsme_test_before=res.rsme(settings.ModelConf.labels.test, apply_inverse=False),
        ame_test=res.ame(settings.ModelConf.labels.test, apply_inverse=True),
        ame_test_before=res.ame(settings.ModelConf.labels.test, apply_inverse=False),
        sme_valid=res.sme(settings.ModelConf.labels.validate, apply_inverse=True),
        sme_valid_before=res.sme(settings.ModelConf.labels.validate, apply_inverse=False),
        rsme_valid=res.rsme(settings.ModelConf.labels.validate, apply_inverse=True),
        rsme_valid_before=res.rsme(settings.ModelConf.labels.validate, apply_inverse=False),
        ame_valid=res.ame(settings.ModelConf.labels.validate, apply_inverse=True),
        ame_valid_before=res.ame(settings.ModelConf.labels.validate, apply_inverse=False),
        stats=str(stats).replace("\n", "\n\t"),
        mean=res.rsme(settings.ModelConf.labels.validate, apply_inverse=True) / stats.loc["mean"].values[0],
        median=res.rsme(settings.ModelConf.labels.validate, apply_inverse=True) / stats.loc["50%"].values[0]
    )
    filename = now.strftime("%Y-%m-%d-%H-%M-%S") + "-" + regression_key + ".txt"
    return filename, content


def logg_result(res, reg_conf, regression_key):
    filename, content = stringify_results(res, reg_conf, regression_key)
    print(content)
    with open(os.path.join(settings.PROJECT_DIR, "logs", filename), "w") as file:
        file.write(content)


def results_as_dict(res):
    train_label = settings.ModelConf.labels.train
    test_label = settings.ModelConf.labels.test
    validate_label = settings.ModelConf.labels.validate

    def reverse_dict(d):
        return {v: k for k, v in d.items()}

    def percentage_error(label, res):
        original = sum(res.original_output(label, True))
        pred = sum(res.prediction(label, True))
        return 100 * np.abs(original - pred) / original

    vdf = res.data(validate_label).copy()
    vdf["prediction"] = res.prediction(validate_label, True)
    vdf["value"] = res.original_output(validate_label, True)
    vdf["abs_error"] = np.abs(vdf["prediction"] - vdf["value"])

    reference_index = ((vdf.year + vdf.month / 12) == (vdf.year + vdf.month / 12).max()).values
    vdf[reference_index].head()
    categ = {}
    for sc in res.datasets.string_cols:
        vdf[sc] = vdf[sc].replace(reverse_dict(res.datasets.category_encoder[sc]))
        temp = vdf.groupby(sc)[["prediction", "value", "abs_error"]].sum()
        temp["percentage_error"] = 100 * temp["abs_error"] / temp["value"]
        categ[sc] = temp.T.to_dict()

    return {
        "model-desc": {
            "lags": [c for c in res.datasets.get_train().columns if "t-" in c]
        },
        "model-performance": {
            train_label: {
                "rsme": res.rsme(train_label, apply_inverse=True),
                "ame": res.ame(train_label, apply_inverse=True),
                "percentage-error": percentage_error(train_label, res)
            },
            test_label: {
                "rsme": res.rsme(test_label, apply_inverse=True),
                "ame": res.ame(test_label, apply_inverse=True),
                "percentage-error": percentage_error(test_label, res)
            },
            validate_label: {
                "rsme": res.rsme(validate_label, apply_inverse=True),
                "ame": res.ame(validate_label, apply_inverse=True),
                "percentage-error": percentage_error(validate_label, res)
            }
        },
        "validation-data-2017": categ
    }
