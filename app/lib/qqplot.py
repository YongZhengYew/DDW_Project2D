import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class QQPlot:
    def __init__(self, df, invFuncParams, MAXWIDTH=4):
        self.MAXWIDTH = MAXWIDTH
        self.columns = df.columns
        self.data = df.to_numpy()
        #self.sortedData = np.apply_along_axis(np.sort, 1, dataDict)
        self.nPlusOne = self.data.shape[0] + 1
        self.invFuncParams = invFuncParams
        self.invFuncs = {
            "exponential": lambda params: lambda x: (-math.log(1-x))/params[0]
        }
    
    def plotAllGraphs(self, columnNames=None, invFuncParams=None):
        fig = plt.figure()
        if columnNames is None:
            columnNames = list(self.columns)
        if invFuncParams is None:
            invFuncParams = self.invFuncParams
        
        count = 0
        for i, columnName in enumerate(columnNames):
            for j, (name, params) in enumerate(invFuncParams.items()):
                count += 1
                print(params, type(params))
                w = min(len(columnNames), self.MAXWIDTH)
                h = max(1, (len(columnNames)*len(invFuncParams) // w)+1)
                subplotParams = (h, w, count)
                invFunc = self.invFuncs[name](params)
                self.plotGraph(columnName, invFunc, name, params, fig, subplotParams)
        plt.show()
    
    def plotGraph(self, columnName, invFunc, invFuncName, params, fig, subplotParams):
        currData = np.sort(self.data[:, self.columns.get_loc(columnName)])

        pairArr = np.empty((currData.shape[0], 2))
        for i, x in enumerate(currData):
            pairArr[i] = [
                invFunc((i+1)/self.nPlusOne),
                x
            ]
        df = pd.DataFrame({
            "data": pairArr[:,0],
            "quantile": pairArr[:,1]
        })
        print(subplotParams)
        ax = fig.add_subplot(*subplotParams)
        sns.scatterplot(
            data=df,
            y="data",
            x="quantile"
        ).set(
            ylabel=columnName,
            xlabel="quantile of " + invFuncName + str(params)
        )




arr = np.random.exponential(1,1000).reshape(10,100)
df = pd.DataFrame({
    "cheese": arr[0,:],
    "wine": arr[1,:],
    "olives": arr[2,:],
    "garum": arr[3,:],
    "vinegar": arr[4,:],
    "fish": arr[5,:],
    "barley": arr[6,:],
    "wheat": arr[7,:],
    "chickpeas": arr[8,:],
    "lard": arr[9,:]
})



q = QQPlot(df, {
    "exponential": (1,)
})
q.plotAllGraphs()