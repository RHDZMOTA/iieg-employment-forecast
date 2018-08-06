from util.serving import prediction_precedure, production_model_procedure, create_data, train_precedure
from optparse import OptionParser


def main():
    parser = OptionParser()
    parser.add_option("--load", action="store_true")
    parser.add_option("--search", action="store_true")
    parser.add_option("--production", action="store_true")
    parser.add_option("--predict", action="store_true")
    kwargs, _ = parser.parse_args(args=None, values=None)

    if getattr(kwargs, "load"):
        create_data()
    if getattr(kwargs, "search"):
        train_precedure()
    if getattr(kwargs, "production"):
        production_model_procedure()
    if getattr(kwargs, "predict"):
        prediction_precedure()


if __name__ == "__main__":
    main()