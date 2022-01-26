#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
"""
File:   assignment01.py
Author: Ishita Trivedi (6893-6496)
Date:   20 September 2021
Desc:   
    
"""


""" =======================  Import dependencies ========================== """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import norm

plt.close('all') #close any open plots

""" ======================  Function definitions ========================== """
l=0.01
def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,x4=None,t4=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data
    if(x4 is not None):
        p4 = plt.plot(x4, t4, 'm') #plot training data

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
   # if(x4 is None):
        #plt.legend((p1[0],p2[0],p3[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0],p4[0]),legend)
              
def evenspace(x,M):
    x3=[]
    arr=np.array_split(x,M)
    for i in range (0,M):
        a=np.median(arr[i])
        x3+=[a]
    return x3

def random1(x,M):
    x3=np.random.choice(x,M)
    x3=np.sort(x3)
    return x3

def create_phi(x,u,M,s):
    P=np.array([x**m for m in range(M+1)])
    P[0]=1
    for i in range(1, M+1):
        P[i]=np.exp((-(x-u[i-1])**2)/(2*(s**2)))
    P=P.T
    return P
    
def train(x,u,M,s):
    P=create_phi(x,u,M,s)
    w = np.linalg.inv(P.T@P+l*(np.identity(M+1)))@P.T@t1
    y = P@w
    return w,y

def test(u,M,w,P):
   # P2=create_phi(data_test,u,M)
    y=P@w
    return y    


def varyM_train():
    n=len(x1)
    s=0.5
    error1=[]
    error2=[]
    for m in range(1, n+1):
        x3= evenspace(x1,m)
        x4= random1(x1,m)
        we,y1=train(x1,x3,m,s)
        P1=create_phi(testData,x3,m,s)
        pred1=test(x3,m,we,P1)
        error1.append(np.mean(np.abs(pred1-testTrue)))
        wr,y2=train(x1,x4,m,s)
        P2=create_phi(testData,x4,m,s)
        pred2=test(x4,m,wr,P2)
        error=Merror_random(x4,m,wr)
        error2.append(np.mean(error))
        fig=plt.figure(figsize=(25,5))
        fig.add_subplot(1,3,1)
        plt.title('x vs t with M= '+ str(m))
        plotData(x1,t1,x2,t2,testData,pred1,testData,pred2,legend=['Training','True','Estimated(EvenSpace)','Estimated(Random)'])
    plt.show()
    return error1,error2

def vary_S():
    s=np.linspace(0.001,10,20)
    m=5
    error1=[]
    error2=[]
    for i in range(0,20):
        x3= evenspace(x1,m)
        x4= random1(x1,m)
        we,y1=train(x1,x3,m,s[i])
        P1=create_phi(testData,x3,m,s[i])
        pred1=test(x3,m,we,P1)
        error1.append(np.mean(np.abs(pred1-testTrue)))
        wr,y2=train(x1,x4,m,s[i])
        P2=create_phi(testData,x4,m,s[i])
        pred2=test(x4,m,wr,P2)
        error=Serror_random(x4,m,wr,s[i])
        error2.append(np.mean(error))
        fig=plt.figure(figsize=(25,5))
        fig.add_subplot(1,3,1)
        plt.title('Varying S with S= '+str(s[i]))
        plotData(x1,t1,x2,t2,testData,pred1,testData,pred2,legend=['Training','True','Estimated(EvenSpace)','Estimated(Random)'])
    plt.show()
    return error1, error2,s
    
# def Mtest_even():
#     error=[]
#     for M in range(1,20):
#         even_means=evenspace(testData, M)
#         P=create_phi(testData, even_means,M,0.5)
#         pred=test(even_means,M,w1[M-1],P)
#         error.append(np.mean(np.abs(pred-testTrue)))
#     return error

def Merror_random(means,M,w):
    error=[]
    for i in range(0,10):
        #ran_means=random1(testData,M)
        P=create_phi(testData, means,M,0.5)
        pred=test(means,M,w,P)
        error.append(np.mean(np.abs(pred-testTrue)))
    return error
            
# def Stest_even(s,M):
#     error=[]
#     for i in range(0,19):
#         even_means=evenspace(testData, M)
#         P=create_phi(testData, even_means,M,s[i])
#         pred=test(even_means,M,w3[M-1],P)
#         error.append(np.mean(np.abs(pred-testTrue)))
#     return error

def Serror_random(means,M,w,s):
    error=[]
    for j in range(0,10):
        P=create_phi(testData,means,M,s)
        pred=test(means,M,w,P)
        error.append(np.mean(np.abs(pred-testTrue)))
    return error

def polynomial_train(x,M):
    X = np.array([x1**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t1
    y = X@w
    return w,y

def polynomial_test(x,w,M):
    X = np.array([x**m for m in range(M+1)]).T
    y = X@w
    return y

def varyM_poly(M):
    error=[]
    for i in range (1,M+1):
        w5,y5=polynomial_train(x1,i)
        y6=polynomial_test(testData,w5,i)
        error.append(np.mean(np.abs(y6-testTrue)))
        fig=plt.figure(figsize=(25,5))
        fig.add_subplot(1,3,1)
        plt.plot(x1,t1,'bo', x2,t2,'g', x1,y5, 'r')
        plt.legend(['Training','True','Estimated'])
        plt.title('Polynomial Regression M= '+ str(i))
    plt.show()
    return error

""" ======================  Variable Declaration ========================== """
#M =  3 #regression model order
#s = 0.5

""" =======================  Load Training Data ======================= """
data_train=np.load('train_data.npy')

x1 = data_train[0,:]
t1 = data_train[1,:]
x_index=np.argsort(x1)
x1=np.sort(x1)
t1=t1[x_index] #for sorting target values with respect to x value indices


x2 = np.arange(-4,3,0.001)
x2 = np.array([i for i in x2 if i<=-1.2 or i>=-0.8])
t2=x2/(x2+1)       #true function
    
""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """


""" ======================== Load Test Data  and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
data_test=np.load('test_data.npy')  
testData=np.sort(data_test)
testTrue=testData/(testData+1)

e1,e2=varyM_train()

e3,e4,sd=vary_S()

e6=varyM_poly(20)

e_sd_m = np.std(e2)
e_sd_s = np.std(e4)
""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """

fig=plt.figure(figsize=(25,5))
fig.add_subplot(1,3,1)
plt.plot(list(range(1,21)),e1,'r', list(range(1,21)),e2,'b')
plt.legend(['Even Error','Random Error'])
plt.title("Errors of RBF while varying M")
# plt.ylim(0,1000)
plt.show()
fig, ax = plt.subplots()
ax.bar(list(range(1,21)), e2, yerr=e_sd_m,align = 'center',alpha = 0.5, ecolor='black', capsize = 10)
plt.title("Random mean Errors vs M with one Standard Deviation")
plt.show()
fig, ax = plt.subplots()
ax.bar(list(range(1,21)), e4, yerr=e_sd_s,align = 'center',alpha = 0.5, ecolor='black', capsize = 10)
plt.title("Random mean Errors vs S with one Standard Deviation")
plt.show()
plt.plot(sd,e3,'r',sd,e4,'b')
plt.legend(['Even Error','Random Error'])
plt.title("Errors of RBF while varying Standard Deviation")
plt.xlim(0.001,10)
plt.show()
plt.title("Errors of RBF vs Polynomial")
plt.plot(list(range(1,21)),e1,'r',list(range(1,21)),e2,'b',list(range(1,21)),e6,'g')
plt.legend(['RBF Even','RBF Random','Poly Error'])
plt.ylim(0,20)
plt.show()



