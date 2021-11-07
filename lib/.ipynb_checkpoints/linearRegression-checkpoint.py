import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

class LinearRegression:
    COLUMNNAME_yGuesses = "yGuesses"
    COLUMNNAME_residuals = "residuals"

    def __init__(self, df, y, learningIncrement):
        self.df = df
        self.xs = [name for name in self.df.columns if name != y] # fact that it is ordered is impt
        self.y = self.df[y]
        self.learningIncrement = learningIncrement

        self.guesses = self.initGuesses()

        #self.incrementMatrix = self.getIncrementMatrix()

    def initGuesses(self) -> pd.DataFrame:
        firstXs = pd.DataFrame([self.df[x].iloc[0] for x in self.xs])
        firstY = self.y.iloc[0]
        lastXs = pd.DataFrame([self.df[x].iloc[-1] for x in self.xs])
        lastY = self.y.iloc[-1]

        gradients = ((lastY-firstY)/(lastXs-firstXs)).iloc[:, 0]
        yIntercepts = (firstY - gradients*firstXs).iloc[:, 0]

        res = []
        for i, x in enumerate(self.xs):
            res.append([gradients[i]*self.df[x] + yIntercepts[i]])
        print(res)
        return pd.DataFrame(res)

    def getIncrementMatrix(self):
        m = self.df.shape[0]
        return (1/(2*m)) * sum((self.guesses - self.y)**2)






df = pd.read_csv("test.csv")

res = LinearRegression(df, "mouse weight", "mouse size")