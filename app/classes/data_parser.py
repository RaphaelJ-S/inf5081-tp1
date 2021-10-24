from .parser_interface import Parser


class Data_Parser(Parser):

    def __init__(self):
        self.data = {}

    def parse(self, str: str):
        array_data = str.strip('\n').split(",")
        self.data[int(array_data[0])] = {
            "RI": float(array_data[1]),
            "Na": float(array_data[2]),
            "Mg": float(array_data[3]),
            "Al": float(array_data[4]),
            "Si": float(array_data[5]),
            "K": float(array_data[6]),
            "Ca": float(array_data[7]),
            "Ba": float(array_data[8]),
            "Fe": float(array_data[9]),
            "Type of glass": int(array_data[10])
        }

    def access_data(self) -> dict:
        return self.data
