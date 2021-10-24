import jupyter
import matplotlib
import numpy
import pandas as pd
import scipy
import sklearn
from classes.reader import Reader
from classes.data_parser import Data_Parser


def main():
    reader = Reader(Data_Parser())
    data = reader.read_data("../glass.data")
    df = pd.DataFrame(data=data).T
    print(df)


if __name__ == "__main__":
    main()
