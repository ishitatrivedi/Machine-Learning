#Ishita  Trivedi
#UFID: 6893-6496
#Assignment 5

import numpy as np
import matplotlib.pyplot as plt
import math
import textwrap
import time

# for plotting
x_steps,y_steps=1,1
y_c,x_c=np.mgrid[slice(-10,10+y_steps,y_steps),
                slice(-10,10+x_steps,x_steps)]

# x-y points with step size 0.25
x_1=np.arange(-10,10,0.25)
y_1=np.arange(-10,10,0.25)
weights=np.array(([1,1,1,1,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,1,1,1,1,1],
              [-5,-3,-1,1,2,5,9,7,3,1,-2,-4,-6,-8]))


def level1_neu(neuron1,neuron2,neuron3,neuron4,bias):
    o_p1=neuron1 - neuron2+bias
    o_p2=neuron3 - neuron4+bias
    o_p=o_p1 - o_p2+bias
    u_nn=np.unique(o_p)
    for i_i in range(6400):
        if o_p[i_i]==np.min(u_nn):
            o_p[i_i]=1
        else:
            o_p[i_i]=0

    u_nn=np.unique(o_p)
    out_2=o_p.reshape(80,80)
    return o_p,out_2,u_nn

def level3_neu(neuron1,neuron2,bias):
    o_p=0.5*neuron1+0.5*neuron2+bias
    u_nn=np.unique(o_p)
    for i_i in range(6400):
        if o_p[i_i]==np.min(u_nn):
            o_p[i_i]=0
        elif o_p[i_i]==np.max(u_nn):
            o_p[i_i]=1
        else:
            o_p[i_i]=-1

    u_nn=np.unique(o_p)
    out_2=o_p.reshape(80,80)
    return o_p,out_2,u_nn

def level2_neu(neuron1,neuron2,bias):
    o_p=0.5*neuron1+0.5*neuron2+bias
    u_nn=np.unique(o_p)
    for i_i in range(6400):
        if o_p[i_i]==np.min(u_nn):
            o_p[i_i]=0
        else:
            o_p[i_i]=1
    u_nn=np.unique(o_p)
    out_2=o_p.reshape(80,80)
    return o_p,out_2,u_nn

arr_neuron=np.array([])
for k_k in range(14):
    arr_n=np.array([])
    for i_i in x_1:
        for j_j in y_1:
            arr_n=np.append(arr_n,(weights[0][k_k]*i_i+weights[1][k_k]*j_j+weights[2][k_k]))
            arr_n=np.heaviside(arr_n,0)
    arr_neuron=np.append(arr_neuron,arr_n)
arr_neuron=np.array(np.split(arr_neuron,14))
print(arr_neuron.shape)

bias2=np.array([0.5])

# for F
af1,afr1,uniquefs1=level1_neu(arr_neuron[4],arr_neuron[5],arr_neuron[10],arr_neuron[13],bias2)
af2,afr2,uniquefs2=level1_neu(arr_neuron[2],arr_neuron[3],arr_neuron[11],arr_neuron[12],bias2)
af3,afr3,uniquefs3=level2_neu(af1,af2,-0.3)
af4,afr4,uniquefs4=level1_neu(arr_neuron[0],arr_neuron[5],arr_neuron[10],arr_neuron[11],bias2)
af5,afr5,uniquefs5=level2_neu(af3,af4,-0.3)

# for U
au1,aur1,uniqueus1=level1_neu(arr_neuron[1],arr_neuron[5],arr_neuron[6],arr_neuron[7],bias2)
au2,aur2,uniqueus2=level1_neu(arr_neuron[0],arr_neuron[1],arr_neuron[6],arr_neuron[9],bias2)
au3,aur3,uniqueus3=level2_neu(au1,au2,-0.3)
au4,aur4,uniqueus4=level1_neu(arr_neuron[9],arr_neuron[8],arr_neuron[5],arr_neuron[1],bias2)
au5,aur5,uniqueus5=level2_neu(au3,au4,-0.3)

a_of_UF,a_r_UF,unique_UF=level3_neu(-au5,af5,-0.8)

plt.figure(figsize=(4,4))
plt.imshow(aur5,extent=[x_c.min(),x_c.max(),y_c.min(),y_c.max()])
plt.figure(figsize=(4,4))
plt.imshow(afr5,extent=[x_c.min(),x_c.max(),y_c.min(),y_c.max()])
plt.figure(figsize=(4,4))
plt.imshow(a_r_UF,extent=[x_c.min(),x_c.max(),y_c.min(),y_c.max()])
plt.show()