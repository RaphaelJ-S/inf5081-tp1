from .parser_interface import Parser


class Reader():

    def __init__(self, parser: Parser):
        self.parser = parser

    def read_data(self, file_path: str) -> dict:
        with open(file_path, "r") as file:
            for line in file:
                self.parser.parse(line)
        return self.parser.access_data()
