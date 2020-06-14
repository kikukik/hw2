# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:44:42 2020

@author: No-Pa
"""
import numpy as np
import matplotlib.pyplot as plt
link1="../hw2/dataSets/densEst1.txt"
link2="../hw2/dataSets/densEst2.txt"
link3="../hw2/dataSets/densEstCombined.txt"

def merge_txt(inp):
    with open('../hw2/dataSets/densEstCombined.txt', 'w') as outfile:
        for fname in inp:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return

def get_lengths():
    l1=0;
    l2=0;
    for line in open(link1):
        l1=l1+1
    for line2 in open(link2):
        l2=l2+1
    return (l1,l2)    

def get_priors(l1,l2):
    p_C1=l1/(l1+l2)
    p_C2=l2/(l1+l2)
    return(p_C1,p_C2)

lengths=get_lengths()
print(get_priors(lengths[0],lengths[1]))
def extract_data(t):
    a=np.empty((0,2),float)
    for line in t:
        v=np.array([[line.split()[0],line.split()[1]]],float)
        a=np.append(a,v,axis=0)
    return a

def get_mean_estimation(points):
    m=np.zeros(2)
    m[0]=sum(points[:,0])
    m[1]=sum(points[:,1])
    m=m/len(points)
    return m

def get_biased_var_estimation(mean_est,points):
    l=points-mean_est
    s=np.zeros((2,2))
    for line in l:
        s=s+np.outer(line,line)
    s=s/len(l)
    return s

def get_unbiased_var_estimation(mean_est,points):
    l=points-mean_est
    s=np.zeros((2,2))
    for line in l:
        s=s+np.outer(line,line)
    s=s/(len(l)-1)
    return s

def print_results(link):
    a=extract_data(open(link))
    mean=get_mean_estimation(a)
    print("mean=",mean)
    sigma=get_unbiased_var_estimation(mean,a)
    print("biased_Sigma=",get_biased_var_estimation(mean,a))
    print("sigma=",sigma)
    return

def gaussian(x,mu,detsigma,sigmainv):
    z=np.array(x-mu)
    enum=np.exp(-0.5*np.dot(np.dot(z,sigmainv),np.transpose(z)))
    denom=np.sqrt(np.power(2*np.pi,len(x))*detsigma)
    return enum/denom

def plot(link):
    a=extract_data(open(link))
    mean=get_mean_estimation(a)
    print("mean=",mean)
    sigma=get_unbiased_var_estimation(mean,a)
    print("biased_Sigma=",get_biased_var_estimation(mean,a))
    print("sigma=",sigma)
    detsigma=np.linalg.det(sigma)
    sigmainv=np.linalg.inv(sigma)
    x=np.linspace(min(a[:,0]),max(a[:,0]),300)
    y=np.linspace(min(a[:,1]),max(a[:,1]),300)
    X,Y=np.meshgrid(x,y)
    Z=np.empty_like(X)
    i=0
    while i<len(X):
        j=0
        while j<len(Y):
            xy=np.array([X[i,j],Y[i,j]])
            ergb=gaussian(xy,mean,detsigma,sigmainv)
            Z[i,j]=ergb
            j=j+1
        i=i+1
    plt.contourf(X,Y,Z,25)
    plt.colorbar()
    plt.scatter(a[:,0],a[:,1],alpha=1,c="white",s=0.8)
    return
def postplot(link):
    g=extract_data(open(link3))
    a=extract_data(open(link))
    prior=len(a)/len(g)
#    print("prior=",prior)
    mean=get_mean_estimation(a)
#    print("mean=",mean)
    sigma=get_unbiased_var_estimation(mean,a)
#    print("biased_Sigma=",get_biased_var_estimation(mean,a))
#    print("sigma=",sigma)
    detsigma=np.linalg.det(sigma)
    sigmainv=np.linalg.inv(sigma)
    x=np.linspace(min(g[:,0]),max(g[:,0]),300)
    y=np.linspace(min(g[:,1]),max(g[:,1]),300)
    X,Y=np.meshgrid(x,y,sparse=False)
    Z=np.empty_like(X)
    i=0
    while i<len(X):
        j=0
        while j<len(Y):
            xy=np.array([X[i,j],Y[i,j]])
            ergb=prior*gaussian(xy,mean,detsigma,sigmainv)
            Z[i,j]=ergb
            j=j+1
        i=i+1
    return [X,Y,Z,prior]

def actual_plotting():
    X,Y,Z1,p1=postplot(link1)
    Z2,p2=postplot(link2)[2:4]
    S=np.add(Z1,Z2)
    print(S)
    Z1neu=np.divide(Z1,S)
    Z2neu=np.divide(Z2,S)
   # for i in np.arange(0,9,1):
  #      for j in np.arange(0,9,1):
    #        print("Z1=", Z1[i,j], "Z1neu=",Z1neu[i,j])
    plt.contour(X,Y,Z1,10)
    plt.contour(X,Y,Z2,10)
    plt.title("likelihood x prior")
    plt.figure()
    plt.contour(X,Y,Z1neu,10)
    plt.colorbar()
    plt.title("p(C1|x)")
    plt.figure()
    plt.contour(X,Y,Z2neu,10)
    plt.colorbar()
    plt.title("p(C2|x)")
    plt.figure()
    dec=np.greater(Z1neu,Z2neu)
    plt.scatter(X,Y,dec)
    plt.title("blue=decideC1, white=decideC2") 
    j=0
    mpoints=np.empty(shape=[0,2])
    while j<len(X):
        i=0
        while i<len(Y):
            if not dec[i,j]:
                xy=np.array([[X[i,j],Y[i,j]]])
                mpoints=np.append(mpoints,xy,axis=0)
                break
            i=i+1
        j=j+1
    plt.figure()
    plt.scatter(mpoints[:,0],mpoints[:,1],s=0.5)
    plt.title("decision boundary") 

    return
actual_plotting()
