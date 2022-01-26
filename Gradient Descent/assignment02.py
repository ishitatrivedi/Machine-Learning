#Submitted by: Ishita Trivedi
#UFID: 6893-6496
#Assignment 2


import pandas as pd
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import numpy as np
import math
import random
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import norm
from statistics import stdev


plt.close('all')
data=np.load('data_set.npz')
# np.shape(data)
# for k in data.keys():
#     print (k)
M=50
s=0.1
learning_rate=0.01

data_train= data['arr_0']               #loading all the datasets
data_val= data['arr_1']
data_test= data['arr_2']

x1 = data_train[:,0]
t1 = data_train[:,1]

x_v = data_val[:,0]
t_v= data_val[:,1]

x_t= data_test[:,0]
t_t=data_test[:,1]

x2 = np.arange(-4,3,0.001)
t2= 3*(x2 + np.sin(x2))*np.exp(-x2**2.0) #true function

t3=3*(x_v+ np.sin(x_v))*np.exp(-x_v**2.0)
t4=3*(x1+ np.sin(x1))*np.exp(-x1**2.0)

mu=x1

def create_phi(x,M,s):                      #creating the feature matrix
    P=np.array([x**m for m in range(M)])
    for i in range(0, M):
        P[i]=np.exp((-(x-mu[i])**2)/(2*(s**2)))
    P=P.T
    return P

def gradient_descent(P,l1,l2,alpha):                #gradient descent function
    #error=[]
    #w = np.linalg.inv(P.T@P)@P.T@t1
    w=np.random.rand(50)
    for i in range (0,1000):
        wn=w-learning_rate*(P.T@(w.T@P.T-t1.T).T + (alpha*l1*np.sign(w)) + (2*alpha*l2*w))
        w=wn
    return wn

def cal_weights(P1):            #calculate weights for all alphas and lambdas
    w2=[]
    for a in range (0,11):
        for l1 in np.arange(0,1,0.01):
            w=gradient_descent(P1,l1,1-l1,a)
            w2 = np.array(np.append(w2,w))
    return w2

def training(P1,weights):               #function for predicting and plotting for train data
    terror=10000
    hyperp=[]
    E_1 = np.array([])
    for p in range(weights.shape[0]):
        alpha = p
        for q in range(weights.shape[1]):
            lambda1 = (q/100)
            Y = P1@weights[p,q]
            if (lambda1==0.4 or lambda1==0.7):
                fig = plt.figure
                fig = plt.figure(figsize=(10,5))
                fig.add_subplot(1,1,1)
                plt.plot(x1,t1, 'bo', x1, Y,'r', x1,t4, 'g')
                plt.legend(['target_train', 'estimated_train','true_train'])
                plt.title('Training    alpha = '+str(p) + ' lambda1 =' + str(lambda1) + ' lambda2 = '+str(1-lambda1))
            E = np.mean(np.abs(Y - t1))
            if E<terror:
                hyperp=[p,lambda1,1-lambda1]
                terror=E
            E_1 = np.append(E_1, E)
    E_1 = np.array(np.split(E_1, 11))
    al = np.array([0,1,2,3,4,5,6,7,8,9,10])
    la = np.linspace(0,99,100)
    a=plt.figure()      
    asp=a.add_subplot(111,projection='3d')
    plt.title('Train')
    X,Y=np.meshgrid(la,al)
    asp.set_xlabel('Lambda')
    asp.set_ylabel('Alpha')
    asp.set_zlabel('Errors')

    
    asp.plot_surface(X,Y,E_1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    return hyperp,E_1

def validation(P2,weights):         #function for predicting and plotting for validation data
    terror=10000
    hyperp=[]
    E_1 = np.array([])
    for p in range(weights.shape[0]):
        alpha = p
        for q in range(weights.shape[1]):
            lambda1 = (q/100)
            Y = P2@weights[p,q]
            if (lambda1==0.4 or lambda1==0.7):
                fig = plt.figure
                fig = plt.figure(figsize=(10,5))
                fig.add_subplot(1,1,1)
                plt.plot(x_v,t_v, 'bo', x_v, Y,'r')
                plt.legend(['target_validate', 'estimated_validate'])
                plt.title('Validation   alpha = '+str(p) + ' lambda1 =' + str(lambda1) + ' lambda2 = '+str(1-lambda1))
            E = np.mean(np.abs(Y - t_v))
            if E<terror:
                hyperp=[p,lambda1,1-lambda1]
                terror=E
            E_1 = np.append(E_1, E)
    E_1 = np.array(np.split(E_1, 11))
    al = np.array([0,1,2,3,4,5,6,7,8,9,10])
    la = np.linspace(0,99,100)
    a=plt.figure()      
    asp=a.add_subplot(111,projection='3d')
    plt.title('Validation')
    X,Y=np.meshgrid(la,al)
    asp.set_xlabel('Lambda')
    asp.set_ylabel('Alpha')
    asp.set_zlabel('Errors')
    asp.plot_surface(X,Y,E_1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    return hyperp,E_1

def test(P3,weights):               #function for predicting and plotting for test data
    terror=10000
    hyperp=[]
    E_1 = np.array([])
    for p in range(weights.shape[0]):
        alpha = p
        for q in range(weights.shape[1]):
            lambda1 = (q/100)
            Y = P3@weights[p,q]
            if (lambda1==0.4 or lambda1==0.7):
                fig = plt.figure
                fig = plt.figure(figsize=(10,5))
                fig.add_subplot(1,1,1)
                plt.plot(x_t,t_t, 'bo', x_t, Y,'r')
                plt.legend(['target_test', 'estimated_test'])
                plt.title('Test    alpha = '+str(p) + ' lambda1 =' + str(lambda1) + ' lambda2 = '+str(1-lambda1))
            E = np.mean(np.abs(Y - t_t))
            if E<terror:
                hyperp=[p,lambda1,1-lambda1]
                terror=E
            E_1 = np.append(E_1, E)
    E_1 = np.array(np.split(E_1, 11))
    al = np.array([0,1,2,3,4,5,6,7,8,9,10])
    la = np.linspace(0,99,100)
    a=plt.figure()
    asp=a.add_subplot(111,projection='3d')
    plt.title('Test')
    X,Y=np.meshgrid(la,al)
    asp.set_xlabel('Lambda')
    asp.set_ylabel('Alpha')
    asp.set_zlabel('Errors')
    asp.plot_surface(X,Y,E_1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    return hyperp,E_1

P1=create_phi(x1,M,s)
P2=create_phi(x_v,M,s)
P3=create_phi(x_t,M,s)
weights=cal_weights(P1)
weights=np.reshape(weights,(11,100,50))

plt.plot(x1,t1,'bo',x2,t2,'g')
plt.title('Train vs True')
plt.legend(['Train','True'])
hyper_t,E_train=training(P1,weights)
hyper_v, E_vali=validation(P2,weights)
hyper_t, E_test=test(P3,weights)
plt.show()


# m1=np.mean(E_train)
# m2=np.mean(E_vali)
# m3=np.mean(E_test)
# sd1=np.std(E_train)
# sd2=np.std(E_vali)
# sd3=np.std(E_test)

# w_a=gradient_descent(P1,0.5,0.5,0.2)
# Y_v=P2@w_a
# plt.plot(x_v,t_v, 'bo', x_v, Y_v,'r')
# plt.legend(['target_validate', 'estimated_validate'])
# plt.title("Alpha=0.5  Lambda=0.1")

# w_a=gradient_descent(P1,0.5,0.5,1)
# Y_v=P1@w_a
# plt.plot(x1,t1, 'bo', x1, Y_v,'r', x1,t4, 'g')
# plt.legend(['target_train', 'estimated_train','true_train'])
# plt.title("Alpha=1  Lambda=0.5")

# w_a=gradient_descent(P1,0.3,0.7,2)
# Y_t=P3@w_a
# plt.plot(x_t,t_t, 'bo', x_t, Y_t,'r')
# plt.legend(['target_test', 'estimated_test'])
# plt.title("Alpha=2  Lambda=0.3")

# w=[]
# for i in range(0,10):
#     w_a=gradient_descent(P1,0.5,0.5,0.1)
#     w=np.append(w,[w_a])
#     Y_v=P2@w_a
#     fig = plt.figure
#     fig = plt.figure(figsize=(10,5))
#     fig.add_subplot(1,1,1)
#     plt.plot(x_v,t_v, 'co', x_v, Y_v,'r', x_v,t3 ,'b')
#     plt.legend(['target_vali', 'estimated_vali','true_vali'])
#     plt.title("Alpha=0.1  Lambda=0.5")