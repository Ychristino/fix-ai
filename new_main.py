# Exemplo de traceback fornecido pelo usuário
from data_iteractor.loader import Loader
from data_iteractor.preProcessor import standardize_format, normalize_log_level, tokenize_code, show_tokens

search_data = False
use_ai = True

python_errors, solutions, public_data = Loader(path='./assets').load_structure()
python_errors = standardize_format(python_errors['traceback'])

traceback_texto = """
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in <module>
TypeError: unsupported operand type(s) for +: 'int' and 'str'
"""
traceback = standardize_format(traceback_texto)

if search_data:
    for error in python_errors.itertuples(index=False):
        print(f'Looking for... {error.name}')
        result = solutions[solutions['error'].str.contains(error.name, case=False, na=False)]
        print(f"First look took out {len(result)} results...")
        if len(result) > 1:
            print(f"Seacrhing for a more specific results for {error.description}...")
            accurate_result = result[result['error'].str.contains(error.description, case=False, na=False)]
            if len(accurate_result) > 0:
                result = accurate_result
                print(f"Found {len(result)} results...")
            else:
                print("Nothing specific")
        print(f"O erro encontrado possivelmente se encontra na linha {error.line_error} do arquivo {error.python_script_error}...")

        if len(result) > 0:
            print("Sugestões:")
            for solution in result.itertuples(index=False):
                print(solution.solution)
        else:
            print('Não foi possível encontrar sugestões anteriores...')

if use_ai:
    tokenized_code = tokenize_code('assets/code_examples/err_01/error_code.py')
    show_tokens(tokenized_code)