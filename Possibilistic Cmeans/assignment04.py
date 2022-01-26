#Submitted by: Ishita Trivedi
#UFID:6893-6496


import numpy as np
import sklearn as sk
import sklearn.datasets as ds
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def data_gen(num_samples, num_features, noClusters1, shuffle=True):
    data=ds.make_blobs(num_samples, num_features, shuffle=False)[0]
    data=data.T
    labels=np.zeros(num_samples)
    j=int(num_samples / noClusters1)
    for i in range(noClusters1):
        p=i*j
        fuzzifier=(i + 1)*j
        labels[p:fuzzifier]=i
    return data, labels

def cluster_verification(data, noClusters1, ccntr, fpart, labels):
    ssd_actual=0
    for i in range(noClusters1):
        data1=data[labels==i]
        fuzzifier=np.mean(data1, axis=0)
        for pt in data1:
            ssd_actual+= np.linalg.norm(pt-fuzzifier)
    clm = np.argmax(fpart, axis=0)
    ssd_clusters=0
    for i in range(noClusters1):
        x2=data[clm==i]
        for pt in x2:
            ssd_clusters+=np.linalg.norm(pt-ccntr[i])
    print("Verify")
    print(ssd_clusters/ssd_actual)

def plotdata(data, ccntr, fpart, noClusters1, labels=None):
    ax=plt.subplots()[1]
    cluster_membership=np.argmax(fpart, axis=0)
    data=PCA(n_components=2).fit_transform(data).T
    for j in range(noClusters1):
        ax.scatter(
            data[0][cluster_membership==j],
            data[1][cluster_membership==j],
            alpha=0.5,label='series '+str(j),edgecolors="none")
    ax.grid(True)
    ax.legend()
    plt.show()

def fcm_criteria(data, ccntr, n, fuzzifier, metric):
    distance=cdist(data.T,ccntr,metric=metric).T
    distance=np.fmax(distance,np.finfo(data.dtype).eps)
    exp=-2./(fuzzifier-1)
    d2=distance**exp
    fpart=d2/np.sum(d2,axis=0,keepdims=1)
    return fpart, distance

def pcm_criteria(data, ccntr, n, fuzzifier, metric):
    distance=cdist(data.T, ccntr, metric=metric)
    distance=np.fmax(distance, np.finfo(data.dtype).eps)
    d2=(distance ** 2)/n
    exp=1. /(fuzzifier - 1)
    d2=d2.T ** exp
    fpart=1. /(1. + d2)
    return fpart, distance

def eta(fpart, distance, fuzzifier):
    fpart=fpart**fuzzifier
    n=np.sum(fpart*distance,axis=1)/np.sum(fpart,axis=1)
    return n

def updateClusters(data, fpart, fuzzifier):
    um=fpart**fuzzifier
    ccntr=um.dot(data.T)/np.atleast_2d(um.sum(axis=1)).T
    return ccntr

def CMEANS(data, noClusters1, fuzzifier, conv_th, maxIterations, criterion_function, metric="euclidean", ccntr0=None, n=None):
    if not data.any() or len(data) < 1 or len(data[0]) < 1:
        print("Data is in in-correct format")
        return
    S, N=data.shape
    if not noClusters1 or noClusters1 <= 0:
        print("Number of clusters must be at least 1")
    if not fuzzifier:
        print("Fuzzifier value must be greater than 1")
        return
    if ccntr0 is None:
        xt=data.T
        ccntr0=xt[np.random.choice(xt.shape[0], noClusters1, replace=False), :]
    ccntr=np.empty((maxIterations, noClusters1, S))
    ccntr[0]=np.array(ccntr0)
    fpart=np.zeros((maxIterations, noClusters1, N))
    itr=0
    while itr<maxIterations - 1:
        fpart[itr], distance=criterion_function(data, ccntr[itr], n, fuzzifier, metric)
        ccntr[itr+1]=updateClusters(data, fpart[itr], fuzzifier)
        # Criteria for stopping the iterations
        if np.linalg.norm(ccntr[itr + 1]-ccntr[itr])<conv_th:
            break
        itr+=1
    return ccntr[itr], ccntr[0], fpart[itr-1], fpart[0], distance, itr

def FCM(data, noClusters1, fuzzifier, conv_th, maxIterations, metric="euclidean", ccntr0=None):
    return CMEANS(data, noClusters1, fuzzifier, conv_th, maxIterations, fcm_criteria, metric, ccntr0=ccntr0)

def PCM(data, noClusters1, fuzzifier, conv_th, maxIterations, metric="euclidean", ccntr0=None):
    ccntr, ccntr0, fpart, fpart0, distance, itr=FCM(data, noClusters1, fuzzifier, conv_th, maxIterations, metric=metric, ccntr0=ccntr0)
    n=eta(fpart, distance, fuzzifier)
    return CMEANS(data, noClusters1, fuzzifier, conv_th, itr, pcm_criteria, metric, ccntr0=ccntr, n=n)

num_samples=3000
num_features=2
noClusters1=3
noClusters2=3
noClusters3=4
fuzzifier=1.2
error=0.001
maxItr=100

iris=ds.load_iris()
labels2=iris.target_names
target2=iris.target
iris=np.array(iris.data).T

digits=ds.load_digits()
labels3=digits.target
digits=np.array(digits.data).T

data1, labels1=data_gen(num_samples, num_features, noClusters1,shuffle=False)

ccntr1, ccntr01, fpart1, fpart01, distance1, itr1=FCM(data1, noClusters1, fuzzifier, error, maxItr)
ccntr2, ccntr02, fpart2, fpart02, distance2, itr2=FCM(iris, noClusters2, fuzzifier, error, maxItr)
ccntr3, ccntr03, fpart3, fpart03, distance3, itr3=PCM(digits, noClusters3, fuzzifier, error, maxItr)

plotdata(data1.T, ccntr1, fpart1, noClusters1)
plotdata(iris.T, ccntr2, fpart2, noClusters2)
plotdata(digits.T, ccntr3, fpart3, noClusters3)
print("Generated dataset")
cluster_verification(data1.T, noClusters1, ccntr1, fpart1, labels1)
print("Iris dataset")
cluster_verification(iris.T, noClusters2, ccntr2, fpart2, target2)
print("Digits dataset")
cluster_verification(digits.T, noClusters3, ccntr3, fpart3, labels3)

#Code referred from https://github.com/holtskinner/PossibilisticCMeans.git