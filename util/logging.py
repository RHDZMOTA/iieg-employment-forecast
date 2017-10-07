from conf import settings
import datetime
import os


def stringify_results(res, reg_conf, regression_key):
    res_string = """

    -------------------------------
    {datetime}

    SELECTED MODEL: {model}

    PRAMETERS:
{params}

    TRAIN DATA
    > SME : {sme_train}
    > RSME: {rsme_train}
    > AME : {ame_train}

    TEST DATA
    > SME : {sme_test}
    > RSME: {rsme_test}
    > AME : {ame_test}

    """

    params = ""
    for param in reg_conf:
        params += "\t> " + param + ": " + str(reg_conf[param]) + "\n"
    now = datetime.datetime.now()
    content = res_string.format(
        datetime=now.strftime("%Y/%m/%d %H:%M:%S"),
        model=regression_key,
        params=params,
        sme_train=res.sme(settings.ModelConf.labels.train),
        rsme_train=res.rsme(settings.ModelConf.labels.train),
        ame_train=res.ame(settings.ModelConf.labels.train),
        sme_test=res.sme(settings.ModelConf.labels.test),
        rsme_test=res.rsme(settings.ModelConf.labels.test),
        ame_test=res.ame(settings.ModelConf.labels.test)
    )
    filename = now.strftime("%Y-%m-%d-%H-%M-%S") + "-" + regression_key + ".txt"
    return filename, content


def logg_result(res, reg_conf, regression_key):
    filename, content = stringify_results(res, reg_conf, regression_key)
    print(content)
    with open(os.path.join(settings.PROJECT_DIR, "logs", filename), "w") as file:
        file.write(content)
