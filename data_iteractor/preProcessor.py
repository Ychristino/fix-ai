from __future__ import annotations

import re
import pandas as pd

from data_iteractor.Exceptions import InvalidInstanceException


def clear_sensitivity_data(message: str | pd.Series):
    regex_get_path = r'(?:[a-zA-Z]:[\\/]|[\\/])[\w\-_.\\/]+'
    regex_python_file = r'\b\w+\.py\b'
    regex_other_files = r'\b\w+\.\w+\b'

    if isinstance(message, str):
        message_return = re.sub(regex_python_file, '[PYTHON_FILE]', message)
        message_return = re.sub(regex_other_files, '[SOME_FILE]', message_return)
        message_return = re.sub(regex_get_path, '[FILE_PATH]', message_return)
        return message_return
    elif isinstance(message, pd.Series):
        message_return = message.replace(regex_python_file, '[PYTHON_FILE]', regex=True)
        message_return = message_return.replace(regex_other_files, '[SOME_FILE]', regex=True)
        message_return = message_return.replace(regex_get_path, '[FILE_PATH]', regex=True)
        return message_return
    else:
        raise InvalidInstanceException(object_type=type(message), expected_type=(str, pd.Series))


def standardize_format(message: str | pd.Series):
    if isinstance(message, str):
        if not message:
            return message

        tokenized_message = tokenize_error_message(message)
        noise_removed = remove_noise(tokenized_message['description']) if tokenized_message['description'] else ''

        message_return = noise_removed.strip()

        if message_return and not message_return[0].isupper():
            message_return = message_return[0].upper() + message_return[1:]

        if not message_return.endswith('.') and message_return != '':
            message_return += '.'

        tokenized_message['description'] = message_return

        return tokenized_message

    elif isinstance(message, pd.Series):
        return pd.json_normalize(message.apply(standardize_format))

    else:
        raise InvalidInstanceException(object_type=type(message), expected_type=(str, pd.Series))


def normalize_log_level(message: str | pd.Series) -> str:
    log_level_mapping = {
        'warn': 'WARNING',
        'info': 'INFO',
        'debug': 'DEBUG',
        'err': 'ERROR',
        'excep': 'EXCEPTION',
        'crit': 'CRITICAL'
    }

    if isinstance(message, str):
        for level in log_level_mapping:
            if level.lower() in message.lower():
                normalized_message = log_level_mapping[level]
                return normalized_message
        return message

    elif isinstance(message, pd.Series):
        return message.apply(normalize_log_level)


def remove_noise(message: str | pd.Series) -> str:
    noise_patterns = [
        r'\[DEBUG\]', # Remove mensagens de debug
        r'\[INFO\]',  # Remove mensagens informativas
        r'\[TRACE\]', # Remove mensagens de rastreamento
        r'ID: \d+',   # Remove IDs de erros
        r'\s+',       # Remove espaços extras
    ]

    if isinstance(message, str):
        cleaned_message = message
        for pattern in noise_patterns:
            cleaned_message = re.sub(pattern, ' ', cleaned_message)
        cleaned_message = cleaned_message.strip()
        return cleaned_message
    elif isinstance(message, pd.Series):
        # Aplicar remoção de ruído a toda a série
        cleaned_message = message.replace(noise_patterns, ' ', regex=True).str.strip()
        return cleaned_message
    else:
        raise InvalidInstanceException(object_type=type(message), expected_type=(str, pd.Series))


def tokenize_error_message(error_message: str):
    # Expressões regulares para capturar diferentes partes da mensagem de erro
    regex_get_line = r'line\s+([0-9]+)'
    regex_get_file_path = r'File\s+"([^"]+\.py)"'
    regex_pthon_script = r'\b\w+\.py\b'
    regex_get_error_name = r'\b(\w+(?:Error|Exception|Warning|Info))\b'
    regex_get_error_description = r'(\b(?:\w+(?:Error|Exception|Warning|Info)):\s)(.*)'
    # Inicializa as variáveis de resultado
    file_path_error = None
    python_file_error = None
    line_error = None
    error_name = None
    error_description = None

    # Extrair o nome do arquivo e o número da linha
    file_path_match = re.search(regex_get_file_path, error_message, re.DOTALL)
    line_match = re.search(regex_get_line, error_message, re.DOTALL)

    if file_path_match:
        file_path_error = file_path_match.group(1)
        python_file_error = re.search(regex_pthon_script, file_path_error).group(0)
    if line_match:
        line_error = line_match.group(1)

    # Extrair o tipo de erro
    error_name_match = re.search(regex_get_error_name, error_message)
    if error_name_match:
        error_name = error_name_match.group(1)

    # Extrair a descrição do erro
    error_description_match = re.search(regex_get_error_description, error_message, re.DOTALL)
    if error_description_match:
        error_description = error_description_match.group(2).strip()

    # Montar o resultado tokenizado
    result = {
        "file_path_error": file_path_error,
        "python_script_error": python_file_error,
        "line_error": line_error,
        "name": error_name,
        "description": error_description
    }

    return result
