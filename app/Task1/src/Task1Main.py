import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ...lib import linearRegression as LR
from ...lib import modelManager as MM
from importlib import resources

class Task1Main:
    def __init__(self):
        print("_______STARTING TASK 1_______")
        self.run()
        print("_______ENDING TASK 1_______")


    def run(self):
        with resources.path("app.Task1.data", "owid-covid-data.csv") as rawData:
            df = pd.read_csv(rawData)

        m = MM.ModelManager()
        m.addDF("df1", df)