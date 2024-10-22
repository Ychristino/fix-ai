import torch

from classifier.Classifier import Classifier
from classifier.BERT import BERT
from data_iteractor.loader import Loader
from data_iteractor.preProcessor import standardize_format, normalize_log_level
import pandas as pd
import torch.nn.functional as F

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# python_errors, solutions, public_data = Loader(path='./assets').load_structure()

# python_errors = standardize_format(python_errors['traceback'])
# python_errors['classified'] = normalize_log_level(python_errors['name'])
# training_dataframe = python_errors[["name", "description", "classified"]]
# print(training_dataframe)

b = BERT()
b.execute()

# Defina seus textos para prever
texts = [
    "invalid syntax",
    "unexpected indent",
    "unsupported operand type(s) for +: 'int' and 'str'",
    "invalid literal for int() with base 10: 'abc'",
    "list index out of range",
    # "'non_existent_key'",
    # "'NoneType' object has no attribute 'attribute'",
    # "No module named 'non_existent_module'",
    # "No module named 'non_existent_module'",
    # "division by zero",
    # "name 'undefined_variable' is not defined",
    # "[Errno 2] No such file or directory: 'non_existent_file.txt'",
    # "[Errno 13] Permission denied: 'restricted_file.txt'",
    # "int too large to convert to float",
    # "EOF when reading a line",
    # "maximum recursion depth exceeded",
    # "maximum recursion depth exceeded while calling a Python object",
    # "The 'imp' module is deprecated",
    # "This feature is deprecated and will be removed in a future version",
    # "Entering debug mode",
    # "Execution took 5 seconds",
]

# Tokenizar os textos
inputs = b.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

# Executar o modelo
with torch.no_grad():
    outputs = b.model(**inputs)

# A saída será as logits
logits = outputs.logits

# Aplicar a função softmax para obter as probabilidades
probabilities = F.softmax(logits, dim=-1)

# Obter as previsões
predictions = torch.argmax(probabilities, dim=-1)

# Iterar sobre as previsões e imprimir cada uma
for i, prediction in enumerate(predictions):
    print(f"Text: '{texts[i]}'")
    print(f"Predicted label: {prediction.item()} - {b.labels.iloc[prediction.item()]} (probabilities: {probabilities[i].tolist()})")
    print()

# messages = """
# Traceback (most recent call last):
#  File "C:\\Users\\Yan\\PycharmProjects\\fix-ai\\main.py", line 15, in <module>
#    python_errors['standarized_message'] = remove_noise(1)
#                                           ^^^^^^^^^^^^^^^
#  File "C:\\Users\\Yan\\PycharmProjects\\fix-ai\\data_iteractor\\preProcessor.py", line 91, in remove_noise
#    raise InvalidInstanceException(object_type=type(message), expected_type=(str, pd.Series))
# data_iteractor.Exceptions.InvalidInstanceException: Object of type int not valid! Expected str or Series.
# """

# standarized_message = standardize_format(messages)

# print(standarized_message)
