import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
rootdir = os.path.dirname(os.path.dirname(currentdir))
print(rootdir)
sys.path.insert(0, rootdir)

from lib import linearRegression as LR
from lib import modelManager as MM

fileURL = "../data/owid-covid-data.csv"
df = pd.read_csv(fileURL)

m = MM.ModelManager()
m.addDF(df)