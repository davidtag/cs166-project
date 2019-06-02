import numpy as np
import pickle
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize
from statsmodels.stats.proportion import proportion_confint
import copy

class data_generator:
    def __init__ (self,n,d,normalize=True):
        """
        n = number data points
        d = data dimensionality
        """
        self.n = n
        self.d = d
        self.normalize = normalize

    def get_dataset(self):
        X = np.random.normal(0,1,(self.d,self.n))
        if self.normalize:
            X = X/np.linalg.norm(X,keepdims=True,axis=0)
        return X

    def get_query(self):
        q = np.random.normal(0,1,(self.d,1))
        if self.normalize:
            q = q/np.linalg.norm(q,keepdims=True,axis=0)
        return q

class dataset:
    def __init__(self,file_features="",path_imgs="",imgsz=(224,224),normalize=True,fname_offt=48):
        self.imgsz = imgsz
        self.path_imgs = path_imgs
        self.fname_offt = fname_offt
        
        #Unpickle
        data = None
        with open(file_features, 'rb') as f:
            data = pickle.load(f)
        
        #Unpack
        self.fnames = data['fnames']
        self.X = data['all_vecs'].T
        if normalize:
            self.X = self.X/np.linalg.norm(self.X,keepdims=True,axis=0)        
        
    def get_img_path(self,idx):
        return self.path_imgs + self.fnames[idx][self.fname_offt:]

    def get_img(self,idx):
        full_name = self.get_img_path(idx)
        img = imread(full_name)
        img = resize(img, self.imgsz, anti_aliasing=True, mode='reflect')
        return img

    def plt_img(self,idx):
        plt.imshow(self.get_img(idx))
        plt.axis('off')
        plt.gca().set_aspect('equal')
        
    def get_features(self,idx):
        return copy.deepcopy(self.X[:,idx:idx+1])

        
def cos_sim(X,q):
    """
    Returns the cosine similarity between each
    column of X and the (single) column of q.
    """
    assert(q.shape[1]==1)
    ip = (q.T@X).flatten()
    norm_x = np.linalg.norm(X,axis=0)
    norm_q = np.linalg.norm(q,axis=0)
    ratio = ip/norm_x/norm_q
    ratio = np.minimum(ratio,1)
    ratio = np.maximum(ratio,-1)
    sim = 1 - np.arccos(ratio)/np.pi
    return sim

def cos_sim_ordered(X,q):
    sims = cos_sim(X,q)
    sims_dec = np.sort(sims)[::-1]
    sims_ranked = np.argsort(sims)[::-1]
    return sims,sims_dec,sims_ranked

def get_weights(k):
    return (1/np.log2(np.arange(k)+2))**0.9

def ndcg(sims,sims_dec,k,c):
    """
    Normalized discounted cumulative gain @ k
    c denotes a list of ranked indices for the
    query q into X. c can contain less than k
    choices, but this only leads to a lower 
    ndcg score.
    """
    assert(len(c)<=k)
    #Gain for Supplied Choices
    w_c = get_weights(len(c))
    s_c = sims[c]
    gain_c = np.inner(100**s_c,w_c)
    
    #Optimal Gain
    w = get_weights(k)
    s_opt = sims_dec[:k]
    gain_opt = np.inner(100**s_opt,w)
    
    #Ratio
    ndcg = gain_c/gain_opt
    return ndcg

def hamming_dist(H_data,H_q):
    assert(H_data.shape[0]==H_q.shape[0])
    assert(H_q.shape[1]==1)
    b = H_data.shape[0]
    return np.sum(1*(H_data-H_q != 0),axis=0)/b

def hamming_dist_ordered(H_data,H_q):
    hdist = hamming_dist(H_data,H_q)
    hdist_inc = np.sort(hdist)
    hdist_ranked = np.argsort(hdist)
    return hdist,hdist_inc,hdist_ranked

def binom(b,probs,complement=False):
    if complement:
        mean = b*(1-probs)
    else:
        mean = b*probs
    std = np.sqrt(b*probs*(1-probs))
    return mean,std

def binom_conf(b,probs,complement=False):
    mean, std = binom(b,probs,complement)
    counts = np.array(mean,dtype=np.int32)
    ci = [_ for _ in proportion_confint(counts,b,0.05,"wilson")]
    return ci

        
    