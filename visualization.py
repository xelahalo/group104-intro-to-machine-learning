import matplotlib.pyplot as plt
import numpy as np

def standard_deviation(X, names):
    plt.figure(figsize=(10,5))

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

def coefficients(V, pcs, n_attr, names):
    plt.figure(figsize=(10, 10))

    legendStrs = ['PC'+str(e+1) for e in pcs]
    bw = .2
    r = np.arange(1,n_attr+1)

    for i in pcs:    
        plt.bar(r+i*bw, V[:,i], width=bw)

    plt.xticks(r+bw, names, rotation=90)
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.title('WPBC: PCA Component Coefficients')

    plt.grid()
    plt.show()

def scatter_pca(Z, y, classes, classNames):
    i, j = 0, 1
    plt.figure()

    for c in range(classes):
        class_mask = y == c
        plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)

    plt.legend(classNames)
    plt.title('WPBC: PCA')
    plt.xlabel('PC{0}'.format(i+1))
    plt.ylabel('PC{0}'.format(j+1))

    plt.show()