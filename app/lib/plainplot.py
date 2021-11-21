import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class PlainPlot:
    def __init__(self, df, yName, xNames=None, FIGSIZE=(16,10), HSPACE=0.4, WSPACE=0.4, MAXWIDTH=4):
        self.FIGSIZE = FIGSIZE
        self.HSPACE = HSPACE
        self.WSPACE = WSPACE
        self.MAXWIDTH = MAXWIDTH

        self.yName = yName
        self.xNames = xNames if xNames is not None else [column for column in df.columns if column != yName]
        self.df = df
        self.funcs = {
            "identity": lambda x: x
        }
    
    def plotAllGraphs(self, xNames=None, funcNames=None):
        fig = plt.figure(figsize=self.FIGSIZE)
        fig.subplots_adjust(hspace=self.HSPACE, wspace=self.WSPACE)
        if xNames is None:
            xNames = self.xNames
        if funcNames is None:
            funcNames = self.funcs.keys()
        
        count = 0
        for i, xName in enumerate(xNames):
            for j, name in enumerate(funcNames):
                count += 1
                w = min(len(xNames), self.MAXWIDTH)
                h = max(1, (len(xNames)*len(funcNames) // w)+1)
                subplotParams = (h, w, count)
                func = self.funcs[name]
                self.plotGraph(xName, func, name, fig, subplotParams)
        plt.show()
    
    def plotGraph(self, xName, func, funcName, fig, subplotParams):
        currData = pd.DataFrame(self.df[xName].apply(func))

        df = currData.join(self.df[self.yName])
        
        ax = fig.add_subplot(*subplotParams)
        sns.scatterplot(
            data=df,
            y=self.yName,
            x=xName
        ).set(
            ylabel=self.yName,
            xlabel=xName
        )




df = pd.read_csv("testMLR.csv")

q = PlainPlot(df, "index")
q.plotAllGraphs()