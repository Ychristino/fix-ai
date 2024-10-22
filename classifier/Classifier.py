import pandas as pd
from sklearn.model_selection import train_test_split

from data_iteractor.loader import Loader, load_execution_errors
from data_iteractor.preProcessor import standardize_format, normalize_log_level


class Classifier:

    def __init__(self, known_errors_list: pd.DataFrame = pd.DataFrame()):
        self.known_errors_list = known_errors_list
        self.model = None
        self.labels = None
    def load_data(self, known_errors_path: str = None):
        python_errors = load_execution_errors(known_errors_path if known_errors_path is not None else './assets')
        python_errors = standardize_format(python_errors['traceback'])
        python_errors['classified'] = normalize_log_level(python_errors['name'])

        self.known_errors_list = python_errors

        return self.known_errors_list

    def prepare_data(self,
                     text_dataframe: pd.DataFrame = None,
                     label_dataframe: pd.DataFrame = None,
                     test_size: int = 0.2,
                     random_state: int = 42
                     ):
        text_dataframe = text_dataframe if text_dataframe is not None else self.known_errors_list
        raw_x_train, raw_x_test, raw_train_labels, raw_test_labels = train_test_split(text_dataframe,
                                                                                      label_dataframe,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state)
        return raw_x_train, raw_x_test, raw_train_labels, raw_test_labels

    def execute(self):
        self.known_errors_list = self.load_data()
        filtered_data = self.known_errors_list[["name", "description", "classified"]]
        self.labels = filtered_data['name']

        raw_x_train, raw_x_test, raw_train_labels, raw_test_labels = self.prepare_data(
            text_dataframe=filtered_data['description'].tolist(),
            label_dataframe=filtered_data['name']
        )
        labels_dict_train = {index: valor for index, valor in enumerate(raw_train_labels)}
        labels_dict_test = {index: valor for index, valor in enumerate(raw_test_labels)}
