import numpy as np
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from initial_data import *

# data = data.to_numpy()
classLabels = data['outcome'].to_numpy()
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(2)))

C = len(classNames)
y_outcome = np.asarray([classDict[label] for label in classLabels])
y_time = data['time'].to_numpy()

X = data.iloc[:, 3:].to_numpy()
X = (X - X.mean()) / X.std()

U,S,V = linalg.svd(X, full_matrices=False)
V = V.T

# variance
rho = (S*S) / (S*S).sum() 

# Project the centered data onto principal component space
Z = X @ V

# print(np.cumsum(rho))
# threshold = 0.99
# plot_rho = rho[:5]
# # Plot variance explained
# plt.figure()
# plt.plot(range(1,len(plot_rho)+1),plot_rho,'x-')
# plt.plot(range(1,len(plot_rho)+1),np.cumsum(plot_rho),'o-')
# plt.plot([1,len(plot_rho)],[threshold, threshold],'k--')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained');
# plt.legend(['Individual','Cumulative','Threshold'])
# plt.grid()
# plt.show()

i = 0
j = 1
f = plt.figure()
plt.title('WPBC: PCA')

for c in range(C):
    class_mask = y_outcome == c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()