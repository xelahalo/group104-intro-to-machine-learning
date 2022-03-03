#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
from scipy.linalg import svd
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')


# ### Importing and visualizing the dataset

# In[2]:


file_path = 'C:/Users/saras/OneDrive/Desktop/DTU/IntroML/Report1/Dataset/wpbc.data'
headers_path ='C:/Users/saras/OneDrive/Desktop/DTU/IntroML/Report1/Dataset/wpbc.names'


# In[3]:


#Loading the data and the headers
wpbc_data = pd.read_csv(file_path, delimiter=',', header=None)
wpbc_headers = pd.read_csv(headers_path, header=None).iloc[:,0].transpose()


# In[4]:


wpbc_data.columns = wpbc_headers


# In[5]:


wpbc_data


# ### Analysis of the data

# In[6]:


wpbc_data.describe()


# In[7]:


wpbc_data.info()


# In[8]:


wpbc_data.isnull().values.any()


# In[9]:


wpbc_data[wpbc_data.lymph_node_status == '?']


# There are four missing values in lymph_node_status, so I'm going to delete them. The other option was to put 0 in those values but we can not say that during the surgery, any lymph node was found.

# In[10]:


wpbc_data = wpbc_data.drop(wpbc_data[wpbc_data.lymph_node_status == '?'].index)


# In[11]:


wpbc_data["lymph_node_status"] = wpbc_data["lymph_node_status"].astype(str).astype(int)


# Change the R and N for 1 or 0

# In[12]:


classDict = {'R':1, 'N':0}
wpbc_data = wpbc_data.replace({'outcome':classDict})


# In[13]:


wpbc_data #check


# In[14]:


wpbc_data.describe()


# Dimension dataset

# In[15]:


dataset = wpbc_data.copy()

y = dataset.iloc[:, 1] #target colum
c = set(y) #different targets in the dataset
X = dataset.iloc[:, 2:] #data columns

N,M = X.shape
print('Number of data objects: ' + str(N))
print('Number of attributes: ' + str(M))
print('Number of classes: ' + str(len(c)) + ' that are ' + str(c))


# Boxplot of attributes - Find outliers

# In[16]:


#Standarising the data
mean = np.mean(X, 0)
std = np.std(X, 0)
X = (X - mean) / std

fig, ax = plt.subplots(figsize=(30,15))
boxplot = sns.boxplot(data=X, ax=ax)
plt.yticks(fontsize=24)
plt.xticks(rotation=90, fontsize=24)


# We can verify that there are not many outliers outside the whiskers, so probably the distribution of our attributes is a normal distribution
# See if the attributes are normally distributed:
# We can see that all attributes are aproximatelly normally distributed except for the time that make sense that doesnt follow a normal distribution.

# In[17]:


#Attributes standard deviation
attributes = X.columns
data = dataset.iloc[:, 2:]
#if i wanted to do it with stadardized data: data = X

n_rows =attributes.size/2
nbins = 40
 
fig, axes = plt.subplots(int(n_rows), 2, figsize=(20, 80))
axes = axes.ravel()

for col, ax in zip(attributes, axes):
    sns.distplot(data[col], bins=nbins, kde=True, ax=ax, kde_kws={"color":"red"})

fig.tight_layout()
plt.show()


# ### Correlation of the attributes: correlation matrix

# In[18]:


#X_correlation = pd.DataFrame(X)
correlationMatrix = X.corr(method="kendall")
correlationMatrix = correlationMatrix.round(2)

fig, ax = plt.subplots(figsize=(30,30),dpi=50)

heatmap = sns.heatmap(correlationMatrix, annot=False, linewidths=.2, ax=ax, cmap="vlag")
heatmap.set_title("Correlation Matrix", fontsize=30)
heatmap.set_xticklabels(attributes, fontsize = 24)
heatmap.set_yticklabels(attributes,fontsize = 24)

plt.show()


# I makes sense that there are more correlation between attributes such as radious, area and perimeter... the radios in different cells are also similar

# ### Variance explained

# In[19]:


dataset = wpbc_data.copy()
X = dataset.iloc[:, 2:] #data columns

#Standarising the data
mean = np.mean(X, 0)
std = np.std(X, 0)
X = (X - mean) / std


# In[20]:


from scipy.linalg import svd

U,S,Vh = svd(X,full_matrices=False)
V = Vh.T

threshold = 0.9
rho = (S*S) / (S*S).sum() 

# Plot variance explained
plt.figure(figsize = (8,5))
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# In[22]:


cumulative = np.cumsum(rho)
principal_comp = 0
for s in cumulative:
    principal_comp = principal_comp +1
    if s>threshold:
        print('We need ' + str(principal_comp) + ' PC to be above the threashold')
        print('The cumulative variance explained of those PC is ' + str(s))
        break


# ### Principal Component Analysis

# In[23]:


classLabels = set(y)
classNames = sorted(set(classLabels))
C = len(classNames)


# In[27]:


Y = X.values #- np.ones((N,1))* X.values.mean(0)


#y = np.asarray([classDict[value] for value in classLabels])
y = dataset.iloc[:, 1]


# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure(figsize = (8,5))
plt.title('2D PCA')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)

plt.legend(['Non-recurrence', 'Recurrence'])
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()


# In[32]:


#3D PCA
# Indices of the principal components to be plotted
i = 0
j = 1
k = 2

# Plot PCA of the data
f = plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')
plt.title('3D PCA', fontsize = 25)

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], Z[class_mask,k], 'o', alpha=.5)
    
plt.legend(['Non-recurrence', 'Recurrence'], fontsize=20)
ax.set_xlabel('PC{0}'.format(i+1))
ax.set_ylabel('PC{0}'.format(j+1))
ax.set_zlabel('PC{0}'.format(k+1))

# Output result to screen
plt.show()


# In[41]:


#Component analysis: PCA component coefficients


# Figure size
f = plt.figure(figsize=(15, 6), dpi=90)
plt.rc('axes', axisbelow=True)

pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)

plt.rcParams["patch.force_edgecolor"] = True

for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw, color=c[i]) #The columns of V gives us the principal component directions
    plt.xlim(0.9,len(V[:,i])+0.4)
    
plt.xticks(r+bw, attributes, rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Attributes', fontsize=12)
plt.ylabel('Component coefficients', fontsize=12)
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()

