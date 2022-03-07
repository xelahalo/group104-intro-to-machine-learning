from attr import attr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

def boxplot_summary(X):
    X = (X - np.mean(X, 0)) / np.std(X, 0)

    _, ax = plt.subplots(figsize=(30,15))
    boxplot = sns.boxplot(data=X, ax=ax)
    plt.yticks(fontsize=24)
    plt.xticks(rotation=90, fontsize=24)
    plt.show()

def standard_deviation(X, names):
    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(bottom=0.35)
    
    r = np.arange(1,X.shape[1]+1)
    plt.bar(r, X.std(0))

    plt.xticks(r, names, rotation=90)
    plt.ylabel('Standard deviation')
    plt.xlabel('Attributes')
    plt.title('WPBC: attribute standard deviations')

    plt.show()

def explain_variance(rho, threshold=0.9, n_comp=5):
    plot_rho = rho[:n_comp]
    plt.figure()

    plt.plot(range(1,len(plot_rho)+1),plot_rho,'x-')
    plt.plot(range(1,len(plot_rho)+1),np.cumsum(plot_rho),'o-')
    plt.plot([1,len(plot_rho)],[threshold, threshold],'k--')

    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual','Cumulative','Threshold'])

    plt.grid()
    plt.show()

def coefficients(V, pcs, n_attr, names, colors, xlabels, showtitle):
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.35)

    labels = ['PC'+str(e+1) for e in pcs]
    bw = .2
    r = np.arange(1,n_attr+1)

    for i in pcs:
        plt.bar(r+(i%3)*bw, V[:,i], width=bw, color=colors[i % 3])

    if(xlabels):
        plt.xlabel('Attributes')
        plt.xticks(r+bw, names, rotation=90)
    else:
        plt.xticks(r+bw, [])

    plt.ylabel('Component coefficients')
    plt.legend(labels)
    if(showtitle):
        plt.title('WPBC: PCA Component Coefficients')
        
    plt.ylim([-0.7, 0.7])

    plt.grid()
    plt.show()

def scatter_pca(Z, y, classes, classNames):
    fig, axs = plt.subplots(5,5, figsize=(20,20))

    for i in range(5):
        for j in range(5):     
            for c in range(classes):
                class_mask = y == c
                ax = axs[i][j]
                ax.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
                ax.set_xlabel('PC{0}'.format(i+1))
                ax.set_ylabel('PC{0}'.format(j+1))
                # ax.legend(classNames)


    fig.suptitle('WPBC: PCA')
    plt.show()

def normal_dist(X):
    attr = X.columns
    n_rows =attr.size/2
    nbins = 20
    
    fig, axes = plt.subplots(int(n_rows), 2, figsize=(20, 80))
    axes = axes.ravel()

    for col, ax in zip(attr, axes):
        sns.distplot(X[col], bins=nbins, kde=True, ax=ax, kde_kws={"color":"red"})

    fig.tight_layout()
    plt.show()

def sturge_rule(N):
    return math.ceil(1 + 3.322 * math.log(N, 10))