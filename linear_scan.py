import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming
import time
import pickle
from imageio import imread
import os
from pylab import get_current_fig_manager
# from sklearn.metrics.pairwise import rbf_kernel


class NN:
  def __init__(self, X, fnames):
    self.X = X
    self.fnames = fnames
    self.d, self.n = X.shape
    self.q = None

  def query_idx(self, i):
    q = X[:,i:i+1]
    self.query(q)

  def query(self, q):
    self.q = q
    _all = self.get_dist()
    # print(_all.mean())
    closest = _all.argsort() #[::-1]
    # idx = [closest[0], closest[1], closest[2], closest[3], closest[-2], closest[1]]
    self.show_query(closest, 3)


  def get_dist(self):
    _all = []
    for i in range(self.n):
    #     ip = np.inner(q[:,0],X[:,i])
        ip = np.linalg.norm(self.q[:,0]-self.X[:,i])
        _all.append(ip)
    _all = np.array(_all)
    return _all


  def show_im(self, i):
    #print(self.X[:,i])
    full_fname = self.fnames[i]
    plt.imshow(imread(full_fname))
    _, fname = os.path.split(full_fname)
    fname, _ = os.path.splitext(fname)
    # plt.title(fname)
    plt.axis('off')


  def show_query(self, closest, N):
    plt.figure(figsize=(8,6))
    thismanager = get_current_fig_manager()
    thismanager.window.wm_geometry("+50+50")
    for i in range(N):
      plt.subplot(N,2,2*i+1)
      self.show_im(closest[i])
      plt.subplot(N,2,2*i+2)
      self.show_im(closest[-(i+1)])

    plt.show()



# fname = "./imnet-100/color_hist-100.p"
fname = "./imnet-val/color_hist-100.p"
fname = "./imnet-val/color_hist-1000.p"
fname = "./imnet-val/hog-1000.p"
with open(fname, 'rb') as f:
  data = pickle.load(f)

fnames = data['fnames']
X = data['all_vecs'].T
# X = X/np.linalg.norm(X,keepdims=True,axis=0)

nn = NN(X, fnames)

for i in range(100):
  nn.query_idx(i)







