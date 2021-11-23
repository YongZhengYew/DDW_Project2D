import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

class LinearRegression:
    def __init__(self, name, df, yName, xNames, learningIncrement, numIter, transformDict=None):
        self.name = name
        self.df = df
        self.yName = yName
        self.xNames = xNames
        self.learningIncrement = learningIncrement
        self.numIter = numIter
        self.transformDict = transformDict

        self.initMatrices()

        self.bestFitY = None
        self.latestTestResult = None
        self.bsHistory = []
        self.costHistory = []
    
    def prepareFeatures(self, df_features):
        res = df_features.to_numpy()
        res = np.insert(res, 0, 1, axis=1)
        return res

    def initBs(self):
        self.bs = np.zeros((self.xs.shape[1], 1))

    def transform(self):
        for columnName, func in self.transformDict.items():
            if columnName == self.yName:
                self.y = np.apply_along_axis(func, 0, self.y)
            elif columnName in self.xNames:
                index = self.xNames.index(columnName)+1 # weird +1 because of initial column of rows
                self.xs = np.insert(self.xs, index+1, np.apply_along_axis(func, 0, self.xs[:, index]), axis = 1)
                self.initBs()
            else:
                raise ValueError("Column name unrecognized!")        

    def initMatrices(self):
        self.xs = self.df[self.xNames]
        self.xs = self.prepareFeatures(self.xs)
        self.y = self.df[[self.yName]].to_numpy()
        self.initBs()
        self.transform()
    
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
        for _ in range(self.numIter):
            self.costHistory.append(compute_cost())
            self.bsHistory.append(self.bs)
            self.bs = incrementBs()
        self.bestFitY = np.matmul(self.xs, self.bs)
 
    def prepareTestSection(self, testSection):
        cols = list(testSection.columns)
        testSection_features = testSection[self.xNames]
        testSection_features = self.prepareFeatures(testSection_features)
        for columnName, func in self.transformDict.items():
            index = cols.index(columnName)+1
            testSection_features = np.insert(testSection_features, index+1, np.apply_along_axis(func, 0, testSection_features[:, index]), axis = 1)
        return testSection_features
    
    def predict(self, testSection):
        testSection = self.prepareTestSection(testSection)
        return np.matmul(testSection, self.bs)

    def getHistories(self):
        return self.bsHistory, self.costHistory
    
    def getJoinedDF(self):
        res = self.df.copy().join(
            pd.Series(self.bestFitY.copy().flatten(), name="BESTFITLINE")
        )
        return res
    
    def rSquared(self):
        varMean = sum((self.y - np.mean(self.y))**2)
        varLine = 0
        for i, y in enumerate(self.y):
            varLine += (self.bestFitY[i] - y)**2
            print(self.bestFitY[i], y)
        return (varMean-varLine)/varMean

if __name__ == "__main__":
    df = pd.read_csv("testSPLR.csv")

    rgs = LinearRegression("test", df, "coffee", ["donuts"], 0.001, 1000, {
        "donuts": lambda x:x**2
    })

    rgs.performRegression()
    df_BFL = rgs.getJoinedDF()
    print(df_BFL)

    sns.scatterplot(
        data=df_BFL,
        y="BESTFITLINE",
        x="donuts"
    )
    print("HELLO")
    print(rgs.rSquared())
    pd.reset_option("display.max_columns")
    pd.reset_option("display.max_rows")

    plt.show()