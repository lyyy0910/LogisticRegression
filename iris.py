import numpy as np
import pandas as pd
from numpy import linalg

df = pd.read_excel('iris.xlsx')
x=np.array([list(df[u'sepal_length']),list(df[u'sepal_width']),list(df[u'petal_length']),list(df[u'petal_width']),np.ones(150)])
str=list(df[u'species'])
tmp=[]
for i in range(x.shape[1]):
    if str[i]=='setosa':
        tmp.append(0)
    if str[i]=='versicolor':
        tmp.append(1)
    if str[i]=='virginica':
        tmp.append(2)
y=np.array(tmp)
tmp=[]
for i in range(x.shape[1]):
    if str[i]=='setosa':
        tmp.append(1)
    else:
        tmp.append(0)
y1=np.array(tmp)
tmp=[]
for i in range(x.shape[1]):
    if str[i]=='versicolor':
        tmp.append(1)
    else:
        tmp.append(0)
y2=np.array(tmp)
tmp=[]
for i in range(x.shape[1]):
    if str[i]=='virginica':
        tmp.append(1)
    else:
        tmp.append(0)
y3=np.array(tmp)

def init():
    tmp=[]
    for i in range(x.shape[0]-1):
        tmp.append([0])
    tmp.append([1])
    beta=np.array(tmp)
    return beta

def sigmoid(x):
    return 1/(1+np.exp(x))

def train(y):
    beta=init()
    for n in range(1000):
        beta_T_x = np.dot(beta.T[0], x)
        dbeta = 0
        d2beta = 0
        for i in range(x.shape[1]):
            dbeta = dbeta - np.dot(np.array([x[:, i]]).T,(y[i] - (1-sigmoid(beta_T_x[i]))))
            d2beta = d2beta + np.dot(np.array([x[:, i]]).T, np.array([x[:, i]])) * (1-sigmoid(beta_T_x[i])) * (1-(1-sigmoid(beta_T_x[i])))
        if np.linalg.det(d2beta)==0:
            break
        beta = beta - np.dot(linalg.inv(d2beta), dbeta)
    return beta


w1=train(y1)
w2=train(y2)
w3=train(y3)
print('三个模型的参数为：')
print(w1)
print(w2)
print(w3)
train_y1=1.0/(1+np.exp(-np.dot(w1.T,x)))
train_y1=train_y1.T
train_y2=1.0/(1+np.exp(-np.dot(w2.T,x)))
train_y2=train_y2.T
train_y3=1.0/(1+np.exp(-np.dot(w3.T,x)))
train_y3=train_y3.T
Y=np.zeros(150)
for i in range(x.shape[1]):
    if train_y1[i-1]>train_y2[i-1] and train_y1[i-1]>train_y3[i-1]:
        Y[i-1]=0
    if train_y2[i - 1] > train_y1[i - 1] and train_y2[i - 1] > train_y3[i - 1]:
        Y[i - 1] = 1
    if train_y3[i - 1] > train_y2[i - 1] and train_y3[i - 1] > train_y1[i - 1]:
        Y[i - 1] = 2
mistake=0
for i in range(x.shape[1]):
    if Y[i]!=y[i]:
        mistake=mistake+1
print('正确率：',1-mistake/150)