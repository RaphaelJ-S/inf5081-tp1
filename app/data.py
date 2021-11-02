import numpy
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing


class Data:

    def __init__(self, raw: "list[list[str]]"):
        self.raw = raw
        self.data_frame = self.toDataFrame(self.cleanup(raw))

    def cleanup(self, raw : "list[list[str]]") -> "list[dict]":
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
        self.data_frame.drop(label, axis=1, inplace=True)

    def splitNormalizedData(self, offset: float):
        return train_test_split(self.getNormalizedData(), self.data_frame["Type of glass"], test_size=offset)

    def splitData(self, offset: float):
        return train_test_split(self.data_frame[self.getFeatures()], self.data_frame["Type of glass"], test_size=offset)

    def getFeatures(self):
        features = list(self.data_frame.keys())
        if features.count("Type of glass") != 0:
            features.pop()
        if features.count("Id") != 0:
            features.pop(0)
        return features

    def getClassData(self, typeOfGlass: int):
        return self.data_frame[self.data_frame["Type of glass"] == typeOfGlass]

    def getNormalizedData(self) -> numpy.array:
        features = self.data_frame[self.getFeatures()]
        std_scale = preprocessing.StandardScaler().fit(features)
        return std_scale.transform(features)

    def toDataFrame(self, data: "list[dict]") -> DataFrame:
        for dataset in data:
            dataset.pop("Id")
        return DataFrame.from_dict(data)

    def resetData(self):
        self.data_frame = self.toDataFrame(self.cleanup(self.raw))

    def __str__(self):
        string = ""
        for set in self.data_frame:
            string += str(set) + '\n'
        return string
