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
import pdb
import matplotlib.gridspec as gridspec

QUERIES_PER_PLOT = 4
PLOTS_PER_QUERY = 12


class NN:
  def __init__(self, X, fnames):
    self.X = X
    self.fnames = fnames
    self.d, self.n = X.shape
    self.q = None
    self.query_count = 0

  def query_idx(self, i):
    q = X[:,i:i+1]
    self.query(q)

  def query(self, q):
    self.q = q
    _all = self.get_dist()
    # print(_all.mean())
    closest = _all.argsort() #[::-1]
    # idx = [closest[0], closest[1], closest[2], closest[3], closest[-2], closest[1]]

    self.query_count += 1
    self.show_query(closest, PLOTS_PER_QUERY)


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
    plt.gca().set_aspect('equal')


  def show_query(self, closest, N):

    cnt = self.query_count % QUERIES_PER_PLOT
    if cnt == 0:
      cnt = QUERIES_PER_PLOT

    if cnt == 1:
      plt.figure(figsize=(15,15))
      thismanager = get_current_fig_manager()
      thismanager.window.wm_geometry("+50+50")

    for i in range(N):
      plt.subplot(QUERIES_PER_PLOT, N, (cnt-1)*N + i+1)
      self.show_im(closest[i])

      # Show anti-examples
      # plt.subplot(2,N, N + i+1)
      # self.show_im(closest[-(i+1)])

    plt.subplots_adjust(wspace=0.001, hspace=0)

    if cnt == QUERIES_PER_PLOT:
      plt.show()



# fname = "./imnet-100/color_hist-100.p"
# fname = "./imnet-val/color_hist-100.p"
# fname = "./imnet-val/color_hist-1000.p"
# fname = "./imnet-val/hog-1000.p"
# fname = "./imnet-val/color_hist-5000.p"
# fname = "./imnet-val/color_hist-10000.p"
fname = "./imnet-val/color_hist-20000.p"
# fname = "./imnet-val/hog-5000.p"
with open(fname, 'rb') as f:
  data = pickle.load(f)

fnames = data['fnames']
X = data['all_vecs'].T
# X = X/np.linalg.norm(X,keepdims=True,axis=0)

nn = NN(X, fnames)
for i in np.random.permutation(len(fnames)):
  nn.query_idx(i)







