import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
from pandas import IntervalIndex
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.preprocessing import label_binarize, KBinsDiscretizer


class Data:

    def __init__(self, raw: "list[list[str]]"):
        self.raw = raw
        self.data_frame = self.toDataFrame(self.cleanup(raw))

    def cleanup(self, raw: "list[list[str]]") -> "list[dict]":
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

    def splitData(self, offset: float):
        return train_test_split(self.data_frame[self.getFeatures()], self.data_frame["Type of glass"], test_size=offset, random_state=42)

    def splitBinarizedData(self, offset: float):
        y = label_binarize(self.data_frame["Type of glass"], classes=[
                           1, 2, 3, 5, 6, 7])
        return train_test_split(self.data_frame[self.getFeatures()], y, test_size=offset, random_state=42)

    def discretizeManual(self, identifier: str,  bounds: list[float]):

        self.data_frame[identifier].update(pd.cut(x=self.data_frame[identifier],
                                                  bins=bounds,
                                                  labels=range(
                                                      len(bounds) - 1),
                                                  ordered=False))

    def discretizeAuto(self, identifier: str, nb_quantiles: int):
        self.data_frame[identifier].update(pd.qcut(
            self.data_frame[identifier], nb_quantiles, labels=range(nb_quantiles)))

    def removeOutliers(self):
        self.data_frame = self.data_frame[(
            np.abs(stats.zscore(self.data_frame)) < 3).all(axis=1)]

    def getFeatures(self):
        features = list(self.data_frame.keys())
        if features.count("Type of glass") != 0:
            features.pop()
        if features.count("Id") != 0:
            features.pop(0)
        return features

    def getClassData(self, typeOfGlass: int):
        return self.data_frame[self.data_frame["Type of glass"] == typeOfGlass]

    def getNormalizedData(self) -> np.array:
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
        return str(self.data_frame)
