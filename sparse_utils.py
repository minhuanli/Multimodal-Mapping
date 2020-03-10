from scipy.io import mmread,mmwrite
import scipy as sp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log,e



# All this preprocess codes are intended for 
# sparse matrix data with form [gene num, cell num]
# Minhuan Li, Dec 2019

def downsample(mtx,M=10000):
    '''
    # mtx is the origin sparse matrix
    # M is the target count number, 10000 here
    # return a sparse matrix in csr format  
    '''

    # csc format, so the indptr gives nonzero elements of every cell (column)
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
    
    return(mtx1.tocsr())


# This Function return the gene i number distribution
# from Sparse matrix mtx2
# The return value is a 1-D array, which is good for next statistic process 

def genedisi(mtx2,i):
    genei = mtx2[i,:]
    genei = genei.todense()
    genei = np.array(genei)
    genei = np.reshape(genei,[-1])
    return(genei)



def musigma_sp(mtx2,axis=1,nn=20):
    '''
    # this function calculate 
    # the mean and standard deviation 
    # for every row of the sparse matrix
    # The parameter nn means the process divide into small groups of nn number 
    # to finish, which decrease the memory required. Or the kernel will crash down
    # mtx2 should better be CSR Format 
    '''
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


def musigma_drop(mtx,mu,sigma):
    
    '''
    drop those genes (row) with mu < 0.05 and sigma < 0.05
    '''
    w = np.where((mu > 0.05) | (sigma > 0.05))
    w = tuple(np.array(w).reshape(-1))
    
    return(mtx[w,:])



def lognorm(mtx):
    a = log(e,2)
    return(mtx.log1p().multiply(a))




def zsn_nn(mtx,mu,sigma):
    '''
    # for all nonzero elements, zsn = (z - mu)/sigma
    # this function is super tricky, about how to subtract a value from the nonzero elements
    # in sparse matrix
    # reference to https://stackoverflow.com/questions/19017804/scipy-sparse-matrix-special-substraction?rq=1
    '''
    mtx1 = mtx.copy()
    # nonzero elements number per row
    nnz_per_row = np.diff(mtx.indptr) 
    
    # duplicate the mu and sigam scalar
    muall = np.repeat(mu, nnz_per_row)
    sigmaall = np.repeat(sigma, nnz_per_row)
    
    # zsn = (z-mu)/sigma
    mtx1.data = (mtx1.data-muall) / sigmaall
    
    return(mtx1,-mu/sigma)



def toarr0(mtx,w,zsn):
    '''
    input: mtx, sparse matrix, [gene num, cell number]
    
    change the data of genes with indices w in the sparse matrix 
    to array; also make the zero point into -mu/sigma, for z-score
    normalization
    
    for z-score check
    
    output: np.array, [gene num,cell_num],

    '''
    
    nw = np.size(w)
    na = np.zeros([nw,mtx.shape[1]])
    test = mtx[w,:]
    for i in range(nw):
        na[i,:] = zsn[w[i]]
    na[test.nonzero()] = test.data
    
    return(na)




def shufflesp_all(mtx,nn=20):
    '''
    shuffle each row of the mtx sparse matrix. 
    nn is batch size, for less memory use
    
    output: shuffled csr matrix
    '''
    
    # step number
    stepnum = np.int( np.ceil( mtx.shape[0]/nn ) )
    
    # use small step, multi times to reduce the memory consumption
    
    for i in tqdm(range(stepnum)):
        
        temp = shufflesp_row(mtx[(i*nn):min([(i+1)*nn,mtx.shape[0]]),:])
        
        if i == 0:
            res = temp
        else:
            res = sp.sparse.vstack([res,temp])
        
    return(res)


def shufflesp_row(mtx):
    '''
    shuffle each row of the mtx sparse matrix. 
    For efficiency, change to lil format first, do shuffle, 
    then change back to csr format
    
    output: shuffled csr matrix
    '''
    mtx1 = mtx.tolil()
    
    ngenes = mtx.shape[0]
    
    index = np.arange(np.shape(mtx)[1])
    
    for i in range(ngenes):
        np.random.shuffle(index)
        mtx1[i,:] = mtx1[i,index]
    
    return(mtx1.tocsr())


def toarr(mtx,w,zsn):
    '''
    input: mtx, sparse matrix, [gene num, cell number]
    
    change the data of cells with indices w in the sparse matrix 
    to array; also make the zero point into -mu/sigma, for z-score
    normalization
    
    output: np.array, [cell_use number, gene num], already transposed for simple_planes.py
    '''
    nw = np.size(w)
    na = np.zeros([mtx.shape[0],nw])
    test = mtx[:,w]
    for i in range(mtx.shape[0]):
        na[i,:] = zsn[i]
    na[test.nonzero()] = test.data
    
    return(na.T)


def plane_cutting(X,trials, cell_sample,cluster_param):
    feature_weights = np.hstack([PCMBK(X,cell_sample,cluster_param) for k in tqdm(range(trials))])
    return feature_weights


def PCMBK(X,cell_sample,k_sub):
    n_cells,n_genes = X.shape
    cell_use = np.random.choice(np.arange(n_cells),cell_sample,replace = False)
    k_sub_i = np.random.randint(k_sub[0],k_sub[1])
    k_guess = KMeans(k_sub_i).fit_predict(X[cell_use,:])
    #k_guess = KMeans(k_sub_i).fit_predict(toarr(X,cell_use,zsn))
    gnout = np.array([one_tree(X,cell_use,k_guess,ikk1,ikk2) for ikk1 in np.unique(k_guess) for ikk2 in np.unique(k_guess) if ikk1<ikk2 ])
    return np.hstack(gnout)


def one_tree(X,cu,k_guess,ik1,ik2,md = 1):
    Xit = X[cu,:]  
    #Xit = toarr(X,cu,zsn)        
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = md)
    XTT = Xit[np.logical_or(k_guess==ik1,k_guess==ik2),:]
    KTT = k_guess[np.logical_or(k_guess==ik1,k_guess==ik2)]
    clf.fit(XTT,KTT)
    feature_pick = np.flatnonzero(clf.feature_importances_)
    return feature_pick



