import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sys
from scipy import sparse
from tqdm import tqdm
from sklearn.cluster import KMeans


##### 
# I modify the following codes for input data 
# X like [gene num, cell num] in sparse matrix form
# Minhuan Li, @ Dec 2019
#### 

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


def plane_cutting(X,trials, cell_sample,cluster_param,zsn):
    feature_weights = np.hstack([PCMBK(X,cell_sample,cluster_param,zsn) for k in tqdm(range(trials))])
    return feature_weights


def PCMBK(X,cell_sample,k_sub,zsn):
    n_genes,n_cells = X.shape
    cell_use = np.random.choice(np.arange(n_cells),cell_sample,replace = False)
    k_sub_i = np.random.randint(k_sub[0],k_sub[1])
    #k_guess = KMeans(k_sub_i).fit_predict(X[cell_use,:])
    k_guess = KMeans(k_sub_i).fit_predict(toarr(X,cell_use,zsn))
    gnout = np.array([one_tree(X,cell_use,k_guess,ikk1,ikk2,zsn) for ikk1 in np.unique(k_guess) for ikk2 in np.unique(k_guess) if ikk1<ikk2 ])
    return np.hstack(gnout)


def one_tree(X,cu,k_guess,ik1,ik2,zsn,md = 1):
    #Xit = X[cu,:]      
    Xit = toarr(X,cu,zsn)        
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = md)
    XTT = Xit[np.logical_or(k_guess==ik1,k_guess==ik2),:]
    KTT = k_guess[np.logical_or(k_guess==ik1,k_guess==ik2)]
    clf.fit(XTT,KTT)
    feature_pick = np.flatnonzero(clf.feature_importances_)
    return feature_pick


# in main() X loads your data matrix

def main():
    for ki in range(5):
        X = np.load(f'/n/home06/minhuan/test/Multi_Model_Mapping/normalized_X/X_noshuffle.npy',allow_pickle=True).all()
        zsn = np.load(f'/n/home06/minhuan/test/Multi_Model_Mapping/normalized_X/z0value_noshuffle.npy')
        iterations = 3000
        n_sub = 5000
        CP = [20,75]
        FW=plane_cutting(X,iterations,n_sub,CP,zsn)
        np.save(f'/n/home06/minhuan/test/Multi_Model_Mapping/normalized_X/Out/gx_{ki}.npy',FW)
    return True

# in main_shuff(), Xs loads your shuffled data matrix

def main_shuff():
    for ki in range(5):
        Xs = np.load(f'/n/home06/minhuan/test/Multi_Model_Mapping/normalized_X/X_shuffle.npy',allow_pickle=True).all()
        zsn = np.load(f'/n/home06/minhuan/test/Multi_Model_Mapping/normalized_X/z0value_noshuffle.npy')
        iterations = 3000
        n_sub = 5000
        CP = [20,75]
        FW=plane_cutting(Xs,iterations,n_sub,CP,zsn)
        np.save(f'/n/home06/minhuan/test/Multi_Model_Mapping/normalized_X/Out/gs_{ki}.npy',FW)
    return True

if __name__ == '__main__':
    main()
    main_shuff()

