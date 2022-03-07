import numpy as np
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from initial_data import *
from visualization import *

classLabels = data['outcome'].to_numpy()
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

C = len(classNames)
y_outcome = np.asarray([classDict[label] for label in classLabels])
y_time = data['time'].to_numpy()
N = len(y_outcome)

# separating the two y vectors from the data
pandasX = data.iloc[:, 3:]
X_ = pandasX.to_numpy()
X = (X_ - X_.mean(0)) / X_.std(0)

U,S,V = linalg.svd(X, full_matrices=False)
V = V.T

# variance
rho = (S*S) / (S*S).sum() 

# Project the centered data onto principal component space
Z = X @ V

standard_deviation(X_, headers[3:])
boxplot_summary(pandasX)
explain_variance(rho, n_comp=30)
scatter_pca(Z, y_outcome, C, classNames)
normal_dist(pandasX)

colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
for i in range(3):
    j = i*3
    coefficients(V, [j, j+1, j+2], X.shape[1], headers[3:], colors[j:j+3], i == 2, i == 0)
    