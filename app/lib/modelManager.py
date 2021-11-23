import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .linearRegression import LinearRegression

class Recipe:
    def __init__(self, name, df, targetName, featureNameList):
        self.name = name
        self.df = df
        self.featureNameList = featureNameList
        self.targetName = targetName
    
    @property
    def combinedNames(self):
        return self.featureNameList + [self.targetName]

class ModelManager:
    def __init__(self):
        self.dfs = {}
        self.recipes = {}
        self.models = {}
    
    def getDF(self, dfName):
        df = self.dfs.get(dfName, None)
        if df is None:
            raise KeyError("DF with name " + dfName + " does not exist!")
        else:
            return df
    
    def addDF(self, name, df):
        self.dfs[name] = df
    
    def getRecipe(self, rceName):
        rce = self.recipes.get(rceName, None)
        if rce is None:
            raise KeyError("FTP with name " + rceName + " does not exist!")
        else:
            return rce
    
    def addRecipe(self, name, dfName, targetName, featureNameList):
        df = self.getDF(dfName)
        self.recipes[name] = Recipe(name, df, targetName, featureNameList)

    def addModel(self, name, rceName, learningIncrement, numIter, randomState=100, testRatio=0.5, tDict=None):
        rce = self.getRecipe(rceName)
        df = rce.df
        yName = rce.targetName
        xNames = rce.featureNameList
        
        testSection, trainSection = self.splitData(
            df[rce.combinedNames],
            randomState,
            testRatio
        )

        newRegression = LinearRegression(name, trainSection, yName, xNames, learningIncrement, numIter, tDict)
        newRegression.performRegression()
        newRegression.predict(testSection)
        self.models[name] = newRegression
    
    def getModel(self, modelName):
        model = self.models.get(modelName, None)
        if model is None:
            raise KeyError("Model with name " + modelName + " does not exist!")
        else:
            return model

    def splitData(self, amalgamation, randomState=100, testRatio=0.5):
        np.random.seed(randomState)
        testIndices = np.random.choice(
            amalgamation.shape[0],
            size=int(testRatio*amalgamation.shape[0]),
            replace=False
        )
        testSection = amalgamation.iloc[testIndices].reset_index(drop=True)
        trainSection = pd.concat([amalgamation, testSection]).drop_duplicates(keep=False).reset_index(drop=True)

        return testSection, trainSection

if __name__ == "__main__":
    m = ModelManager()
    m.addDF("testdf1", pd.read_csv("testMLR.csv"))
    m.addRecipe("testrce1", "testdf1", "coffee", ["donuts"])

    for i in range(10):
        m.addModel(
            "testModel_"+str(i),
            "testrce1",
            0.001,
            1000,
            100,
            0.2,
            {
                "donuts": lambda x:x**2
            }
        )

    res = m.getModel("testModel_1").getJoinedDF()
    sns.scatterplot(
        data=res,
        y="BESTFITLINE",
        x="donuts"
    )
    plt.show()
