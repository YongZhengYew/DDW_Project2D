import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

class LinearRegression:
    def __init__(self, df, y, learningIncrement, n):
        self.df = df

        self.xs = self.df[[column for column in self.df.columns if column != y]]
        self.xs = self.xs.to_numpy()
        self.xs = np.insert(self.xs, 0, 1, axis=1)

        self.y = self.df[[y]].to_numpy().reshape(len(self.xs), 1)

        self.learningIncrement = learningIncrement

        self.bs = np.ones((self.xs.shape[1], 1))

        print(self.xs)
        print(self.y)
        print(self.bs)

        for _ in range(n):
            self.bs = self.incrementBs()
            print(self.bs)


    def incrementBs(self):
        return self.bs - self.learningIncrement*(1/len(self.xs))*(np.matmul(
            self.xs.transpose(),
            np.matmul(
                self.xs,
                self.bs
            ) - self.y
        ))
    
    def plotBsLine(self):
        self.bestFitY = np.matmul(self.xs, self.bs)
        res = self.df.copy()
        res = res.join(pd.Series(self.bestFitY.flatten(), name="BESTFITLINE"))
        return res


df = pd.read_csv("testMLR.csv")

rgs = LinearRegression(df, "deliciousness", 0.001, 1000)

df_BFL = rgs.plotBsLine()
print(df_BFL)

sns.scatterplot(
    data=df_BFL,
    y="BESTFITLINE",
    x="index"
)

sns.scatterplot(
    data=df_BFL,
    y="deliciousness",
    x="index"
)

plt.show()