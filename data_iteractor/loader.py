import os
import pandas as pd

from data_iteractor.Exceptions import NoSuchFileException


def load_execution_errors(path: str):
    if os.path.exists(os.path.join(path, 'erros_python.json')):
        return pd.read_json(os.path.join(path, 'erros_python.json'))
    else:
        raise NoSuchFileException(path=path, filename='erros_python.json')


def load_fixes(path: str):
    if os.path.exists(os.path.join(path, 'solucoes_python.json')):
        return pd.read_json(os.path.join(path, 'solucoes_python.json'))
    else:
        raise NoSuchFileException(path=path, filename='solucoes_python.json')


def load_public_data(path: str):
    if os.path.exists(os.path.join(path, 'fontes_dados_publicos.json')):
        return pd.read_json(os.path.join(path, 'fontes_dados_publicos.json'))
    else:
        raise NoSuchFileException(path=path, filename='fontes_dados_publicos.json')


class Loader:
    def __init__(self, path: str, execution_errors: pd.DataFrame = None, solutions: pd.DataFrame = None, public_data: pd.DataFrame = None):
        self.path = path
        self.execution_errors = execution_errors
        self.public_data = public_data
        self.solutions = solutions

    def load_structure(self):
        self.execution_errors = load_execution_errors(path=self.path)
        self.public_data = load_public_data(path=self.path)
        self.solutions = load_fixes(path=self.path)

        return self.execution_errors, self.solutions, self.public_data
