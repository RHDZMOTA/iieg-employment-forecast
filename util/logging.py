from conf import settings
import pandas as pd
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
        mean=res.rsme(settings.ModelConf.labels.validate, apply_inverse=True)/stats.loc["mean"].values[0],
        median=res.rsme(settings.ModelConf.labels.validate, apply_inverse=True)/stats.loc["50%"].values[0]
    )
    filename = now.strftime("%Y-%m-%d-%H-%M-%S") + "-" + regression_key + ".txt"
    return filename, content


def logg_result(res, reg_conf, regression_key):
    filename, content = stringify_results(res, reg_conf, regression_key)
    print(content)
    with open(os.path.join(settings.PROJECT_DIR, "logs", filename), "w") as file:
        file.write(content)
