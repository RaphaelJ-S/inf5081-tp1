from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self, raw: list[list[str]]):
        self.raw = raw
        self.data = self.cleanup(raw)

    def cleanup(self, raw) -> list[dict]:
        clean = []
        for list in raw:
            clean.append({
                "Id": int(list[0]),
                "RI": float(list[1]),
                "Na": float(list[2]),
                "Mg": float(list[3]),
                "Al": float(list[4]),
                "Si": float(list[5]),
                "K": float(list[6]),
                "Ca": float(list[7]),
                "Ba": float(list[8]),
                "Fe": float(list[9]),
                "Type of glass": int(list[10])
            })
        return clean

    def removeColumn(self, label: str):
        for set in self.data:
            set.pop(label)

    def splitData(self, offset: float) -> list[list[dict]]:
        return train_test_split(self.data, test_size=offset, random_state=42)

    def sliceLabel(self):
        label = []
        data = []
        for dataset in self.data:
            data.append(self.sliced_dataset(dataset))
            label.append(dataset["Type of glass"])
        return [data, label]

    def toDataFrame(self) -> DataFrame:
        for dataset in self.data:
            dataset.pop("Id")
        return DataFrame.from_dict(self.data)

    def sliced_dataset(self, dataset: dict):
        sliced_data = []
        if not (dataset.get("RI") is None):
            sliced_data.append(dataset["RI"])
        if not (dataset.get("Na") is None):
            sliced_data.append(dataset["Na"])
        if not (dataset.get("Mg") is None):
            sliced_data.append(dataset["Mg"])
        if not (dataset.get("Al") is None):
            sliced_data.append(dataset["Al"])
        if not (dataset.get("Si") is None):
            sliced_data.append(dataset["Si"])
        if not (dataset.get("K") is None):
            sliced_data.append(dataset["K"])
        if not (dataset.get("Ca") is None):
            sliced_data.append(dataset["Ca"])
        if not (dataset.get("Ba") is None):
            sliced_data.append(dataset["Ba"])
        if not (dataset.get("Fe") is None):
            sliced_data.append(dataset["Fe"])
        return sliced_data

    def resetData(self):
        self.data = self.cleanup(self.raw)

    def __str__(self):
        string = ""
        for set in self.data:
            string += str(set) + '\n'
        return string
