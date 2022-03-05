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
X_ = data.iloc[:, 3:].to_numpy()
X = (X_ - X_.mean(0)) / X_.std(0)

U,S,V = linalg.svd(X, full_matrices=False)
V = V.T

# variance
rho = (S*S) / (S*S).sum() 

# Project the centered data onto principal component space
Z = X @ V

standard_deviation(X_, headers[3:])
explain_variance(rho, n_comp=30)
scatter_pca(Z, y_outcome, C, classNames)
for i in range(3):
    j = i*3
    coefficients(V, [j, j+1, j+2], X.shape[1], headers[3:])
    