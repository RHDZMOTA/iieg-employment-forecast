import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_FOLDER = os.environ.get("DATA_FOLDER")
CONF_FOLDER = os.environ.get("CONF_FOLDER")
RAW_DATA = os.environ.get("RAW_DATA")
DATASETS = os.environ.get("DATASETS")
DROPBOX_DOWNLOAD = os.environ.get("DROPBOX_DOWNLOAD")
INSURED_EMPLOYMET_CSV = os.environ.get("INSURED_EMPLOYMET_CSV")
INSURED_EMPLOYMET_PICKLE = os.environ.get("INSURED_EMPLOYMET_PICKLE")
TRAIN_LABEL = os.environ.get("TRAIN_LABEL")
TEST_LABEL = os.environ.get("TEST_LABEL")
VALIDATE_LABEL = os.environ.get("VALIDATE_LABEL")
CONNECTION_STRING= os.environ.get("CONNECTION_STRING")


GENERIC_CSV_FILENAME = os.environ.get("GENERIC_CSV_FILENAME")
GENERIC_PICKLE_FILENAME = os.environ.get("GENERIC_PICKLE_FILENAME")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DATA_PATH = join(PROJECT_DIR, DATA_FOLDER)
RAW_DATA_PATH = join(DATA_PATH, RAW_DATA)
DATASETS_PATH = join(DATA_PATH, DATASETS)


class DataFilesConf:

    class Paths:
        data = DATA_PATH
        conf = join(PROJECT_DIR, CONF_FOLDER)
        raw_data = RAW_DATA_PATH
        datasets = DATASETS_PATH

    class FileNames:
        generic_filename_csv = join(DATA_PATH, GENERIC_CSV_FILENAME)
        insured_employment_csv = join(RAW_DATA_PATH, INSURED_EMPLOYMET_CSV)
        insured_employment_pickle = join(RAW_DATA_PATH, INSURED_EMPLOYMET_PICKLE)


class ModelConf:

    class labels:
        train = TRAIN_LABEL
        test = TEST_LABEL
        validate = VALIDATE_LABEL