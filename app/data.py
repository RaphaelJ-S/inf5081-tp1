import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize


class Data:
    """
    Cette classe fait la transformation des données brutes vers des DataFrame et définie les différentes
    opération permises sur ces données. À Noter que ces opérations sont définies pour les données du 
    fichier 'glass.data'.
    """

    def __init__(self, raw: "list[list[str]]"):
        """
        Prend les données sous forme de listes de liste de string et les stock dans @data_frame. 
        Les données originales sont gardée dans @raw
        """
        self.raw = raw
        self.data_frame = self.toDataFrame(self.cleanup(raw))

    def cleanup(self, raw: "list[list[str]]") -> "list[dict]":
        """
        Transforme les données brutes en liste de dictionnaires contenants l'identifiant
        , les attributs et la classe pour chaque chaque enregistrement.
        """
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
        """Supprime la colonne @label des données"""
        self.data_frame.drop(label, axis=1, inplace=True)

    def splitData(self, offset: float):
        """
        Retourne les données séparées en ensembles d'entrainement et de test ainsi que leurs labels respectifs
        @offset défini la taille de l'ensemble de test. Doit se trouver entre 0 et 1.
        """
        return train_test_split(self.data_frame[self.getFeatures()], self.data_frame["Type of glass"], test_size=offset, random_state=42)

    def splitBinarizedData(self, offset: float):
        """
        Retourne les données séparées en ensembles d'entrainement et de test ainsi que leurs labels respectifs
        binarisés.
        @offset défini la taille de l'ensemble de test. Doit se trouver entre 0 et 1.
        """
        y = label_binarize(self.data_frame["Type of glass"], classes=[
                           1, 2, 3, 5, 6, 7])
        return train_test_split(self.data_frame[self.getFeatures()], y, test_size=offset, random_state=42)

    def discretizeManual(self, identifier: str,  bounds: "list[float]"):
        """
        Discrétise les données de la colonne @identifier selon les bornes @bounds.
        """
        self.data_frame[identifier].update(pd.cut(x=self.data_frame[identifier],
                                                  bins=bounds,
                                                  labels=range(
                                                      len(bounds) - 1),
                                                  include_lowest=True,
                                                  ordered=False))

    def discretizeAuto(self, identifier: str, nb_quantiles: int):
        """
        Discrétise les données de la colonne @identifier selon un nombre de quantiles @nb_quantiles
        """
        self.data_frame[identifier].update(pd.qcut(
            self.data_frame[identifier], nb_quantiles, labels=range(nb_quantiles)))

    def removeOutliers(self):
        """
        Élimine tous les outliers des données
        """
        self.data_frame = self.data_frame[(
            np.abs(stats.zscore(self.data_frame)) < 3).all(axis=1)]

    def getFeatures(self):
        """
        Retourne le nom des attributs sans l'Id la classe.
        """
        features = list(self.data_frame.keys())
        if features.count("Type of glass") != 0:
            features.pop()
        if features.count("Id") != 0:
            features.pop(0)
        return features

    def getClassData(self, typeOfGlass: int):
        """
        Retourne toutes les données de la classe @typeOfGlass
        """
        return self.data_frame[self.data_frame["Type of glass"] == typeOfGlass]

    def getNormalizedData(self) -> np.array:
        """
        Retourne une copie des données normalisées
        """
        features = self.data_frame[self.getFeatures()]
        std_scale = preprocessing.StandardScaler().fit(features)
        return std_scale.transform(features)

    def toDataFrame(self, data: "list[dict]") -> DataFrame:
        """
        Transforme la liste de dictionnaire @data en DataFrame et enlève l'attribut Id.
        """
        for dataset in data:
            dataset.pop("Id")
        return DataFrame.from_dict(data)

    def resetData(self):
        """
        Modifie les données pour les retourner dans leurs état original, sans Id
        """
        self.data_frame = self.toDataFrame(self.cleanup(self.raw))

    def __str__(self):
        return str(self.data_frame)
