import tokenize
from io import BytesIO


class CodeTokenizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tokens = []

    def load_code(self):
        with open(self.file_path, 'r', encoding='utf-8') as arquivo:
            return arquivo.read()

    def tokenize_code(self):
        codigo = self.load_code()
        codigo_bytes = BytesIO(codigo.encode('utf-8')).readline

        for token in tokenize.tokenize(codigo_bytes):
            if token.type != tokenize.ENCODING and token.type != tokenize.NEWLINE:
                self.tokens.append((token.type, token.string))
        return self.tokens

    def show_tokens(self):
        for tipo, valor in self.tokens:
            print(f"Tipo: {tipo}, Valor: '{valor}'")


# Exemplo de uso
if __name__ == "__main__":
    caminho_arquivo = "assets/code_examples/err_01/error_code.py"  # Substitua pelo caminho do arquivo desejado
    tokenizador = CodeTokenizer(caminho_arquivo)
    tokenizador.tokenize_code()
    tokenizador.show_tokens()
