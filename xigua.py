import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg

data=pd.read_excel('xigua.xlsx')

x=np.array([list(data[u'密度']),list(data[u'含糖率']),[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

y=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

def init(): #初始化w,b
    tmp=[]
    for i in range(len(x)-1):
        tmp.append([0])
    tmp.append([1])
    beta=np.array(tmp)
    return beta

def sigmoid(x):
    return 1/(1+np.exp(x))

def train():
    beta=init()
    for n in range(10000):
        beta_T_x=np.dot(beta.T[0],x) #wx+b
        dbeta=0 #beta一阶导数
        d2beta=0 #d2beta二阶导数
        for i in range(x.shape[1]): #x.shape[1]表示列数
            dbeta=dbeta-np.dot(np.array([x[:,i]]).T,(y[i]-(1-sigmoid(beta_T_x[i]))))
            d2beta=d2beta+np.dot(np.array([x[:,i]]).T,np.array([x[:,i]]))*(1-sigmoid(beta_T_x[i]))*sigmoid(beta_T_x[i])
        beta = beta - np.dot(linalg.pinv(d2beta), dbeta)
    return beta

print(train())
beta=train().T
train_y=1.0/(1+np.exp(-np.dot(train().T,x)))
train_y=train_y.T
test=np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
for i in range(x.shape[1]):
    if train_y[i-1] >= test[i-1]:
        train_y[i-1]=1
    else: train_y[i-1]=0
mistake=0
for i in range(x.shape[1]):
    if train_y[i-1]!=y[i-1]:
        mistake=mistake+1
print('正确率：',1-mistake/17)
print('函数: y = ',beta[0][0],'x1 + ',beta[0][1],'x2 + ',beta[0][2])
for i in range(17):
    if y[i]==1:
        plt.plot(x[0,i], x[1,i],'r-o')
    else:
        plt.plot(x[0,i], x[1,i],'g-o')
X=np.linspace(0,1,200)
Y=-train()[0]/train()[1]*X-train()[2]/train()[1]
plt.plot(X,Y)
plt.show()