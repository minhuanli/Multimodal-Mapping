from scipy.io import mmread,mmwrite
import scipy as sp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log,e

# mtx is the origin sparse matrix
# M is the target count number, 10000 here
# return a sparse matrix in csc format  
def downsample(mtx,M=10000):

    # csc format, so the indptr gives nonzero elements of every cell
    mtx1 = mtx.tocsc()

    # drop cells with read < 10000
    genect = np.sum(mtx1,axis=0)
    genect = np.array(genect)[0]
    w = np.where(genect > M)
    mtx1 = mtx1[:,tuple(np.array(w).reshape(-1))]

    # index number for nonzero element in mtx1.data
    nnindex = mtx1.indptr
    
    # NN, row (cell) number
    NN = mtx1.shape[1]
    
    # all nonzero elements of the matrix, a 1D array
    allnn = mtx1.data
    
    # do the random selection
    for i in tqdm(range(NN)):
        
        nn = nnindex[i+1]-nnindex[i] # nonzero elements number for cell i
        
        temp = np.zeros(nn,dtype=np.int16)
        
        ori = allnn[nnindex[i]:nnindex[i+1]]  # origin data
        
        weights = ori/ori.sum()
        
        c = np.random.choice(np.arange(nn),size=M,p=weights.reshape(-1))
        
        for j in c:
            temp[j] += 1   # downsampled data
        
        allnn[nnindex[i]:nnindex[i+1]] = temp
    
    mtx1.data = allnn
    
    return(mtx1)


# This Function return the gene i number distribution
# from Sparse matrix mtx2
# The return value is a 1-D array, which is good for next statistic process 

def genedisi(mtx2,i):
    genei = mtx2[i,:]
    genei = genei.todense()
    genei = np.array(genei)
    genei = np.reshape(genei,[-1])
    return(genei)



# this function calculate 
# the mean and standard deviation 
# for every row of the sparse matrix
# The parameter nn means the process divide into small groups of nn number 
# to finish, which decrease the memory required. Or the kernel will crash down
# mtx2 should better be CSR Format 
def musigma_sp(mtx2,axis=1,nn=20):
    
    # create array for E(X)2 and E(X2)
    mu = np.zeros( mtx2.shape[1-axis] )
    sqrmu = np.zeros( mtx2.shape[1-axis] )
    
    # step number
    stepnum = np.int( np.ceil( mtx2.shape[1-axis]/nn ) )
    
    # use small step, multi times to reduce the memory consumption
    for i in tqdm(range(stepnum)):
        
        # E(X)2
        tempmu = np.mean(mtx2[(i*nn):min([(i+1)*nn,mtx2.shape[1-axis]]),:],axis=axis)
        tempmu = np.array(tempmu).reshape(-1)
        mu[(i*nn) : min([(i+1)*nn,mtx2.shape[1-axis]])]= tempmu  # E(X)
        
        # E(X2)
        sqrmtx2 = mtx2[(i*nn):min([(i+1)*nn,mtx2.shape[1-axis]]),:].copy()
        sqrmtx2.data **= 2
        temp = np.mean(sqrmtx2,axis=axis)
        sqrmu[(i*nn) : min([(i+1)*nn,mtx2.shape[1-axis]])] = np.array(temp).reshape(-1)  # E(X2)
    
    var = sqrmu - mu**2  # var = E(X2) - E(X)2
    sigma = np.sqrt(var)   # standard deviation
    
    return(mu,sigma)


def lognorm(mtx):
    a = log(e,2)
    return(mtx.log1p().multiply(a))



def zsn_nn(mtx,mu,sigma):
    mtx1 = mtx.copy()
    # nonzero elements number per row
    nnz_per_row = np.diff(mtx.indptr) 
    
    # duplicate the mu and sigam scalar
    muall = np.repeat(mu, nnz_per_row)
    sigmaall = np.repeat(sigma, nnz_per_row)
    
    # zsn = (z-mu)/sigma
    mtx1.data = (mtx1.data-muall) / sigmaall
    
    return(mtx1)



