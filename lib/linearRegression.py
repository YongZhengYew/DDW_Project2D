import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

class LinearRegression:
    def __init__(self, df, yName, learningIncrement, numIter):
        self.df = df
        self.yName = yName
        self.learningIncrement = learningIncrement
        self.numIter = numIter

        self.initAllParams()

        self.bsHistory = []
        self.costHistory = []
    
    def initAllParams(self):
        def initXs():
            self.xs = self.df[[column for column in self.df.columns if column != self.yName]]
            self.xs = self.xs.to_numpy()
            self.xs = np.insert(self.xs, 0, 1, axis=1)
        def initY():
            self.y = self.df[[self.yName]].to_numpy()
        def initBs():
            self.bs = np.ones((self.xs.shape[1], 1))
        initXs()
        initY()
        initBs()

    def transform(self, dictColumnNames):
        for oldColumn, (func, newColumn) in dictColumnNames.items():
            self.df[newColumn] = self.df[oldColumn].apply(func)
        self.initAllParams()
    
    def performRegression(self):
        def compute_cost():
            yHat = np.matmul(self.xs, self.bs)
            m = self.y.shape[0]
            cost = (1/(2*m)) * sum(
                np.matmul(
                    (yHat-self.y).transpose(),
                    (yHat-self.y)
                )
            )
            return cost
        def incrementBs():
            return self.bs - self.learningIncrement*(1/len(self.xs))*(np.matmul(
                self.xs.transpose(),
                np.matmul(
                    self.xs,
                    self.bs
                ) - self.y
            ))
        def plotBsLine():
            self.bestFitY = np.matmul(self.xs, self.bs)
            res = self.df.copy()
            res = res.join(pd.Series(self.bestFitY.flatten(), name="BESTFITLINE"))
            return res

        for _ in range(self.numIter):
            self.costHistory.append(compute_cost())
            self.bsHistory.append(self.bs)
            self.bs = incrementBs()
        return plotBsLine()
    
    def showHistory(self):
        print(self.costHistory)
        print("______________________________________")
        print(self.bsHistory)


df = pd.read_csv("testSPLR.csv")

rgs = LinearRegression(df, "coffee", 0.001, 1000)
rgs.transform({
    "donuts": (lambda x:x**2, "donutsSquared")
})
df_BFL = rgs.performRegression()
rgs.showHistory()

sns.scatterplot(
    data=df_BFL,
    y="BESTFITLINE",
    x="donuts"
)

pd.reset_option("display.max_columns")
pd.reset_option("display.max_rows")

plt.show()