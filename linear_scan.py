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
from skimage.transform import resize

QUERIES_PER_PLOT = 8
#PLOTS_PER_QUERY = QUERIES_PER_PLOT*2
PLOTS_PER_QUERY = 16
IMSIZE = (224, 224)


class NN:
  def __init__(self, X, fnames):
    self.X = X
    self.fnames = fnames
    self.d, self.n = X.shape
    self.q = None
    self.query_count = 0
    self.qidx = None

  def query_idx(self, i):
    q = X[:,i:i+1]
    self.qidx = i
    self.query(q)

  def query(self, q):
    t1 = time.time()
    self.q = q
    _all = self.get_dist()
    # print(_all.mean())
    closest = _all.argsort() #[::-1]
    # idx = [closest[0], closest[1], closest[2], closest[3], closest[-2], closest[1]]

    self.query_count += 1
    t2 = time.time()
    print("query took ", t2 - t1, " seconds")
    self.show_query(closest, PLOTS_PER_QUERY)


  def get_dist(self):
    t1 = time.time()

    _all = []
    for i in range(self.n):
    #     ip = np.inner(q[:,0],X[:,i])
        ip = np.linalg.norm(self.q[:,0]-self.X[:,i])
        _all.append(ip)
    _all = np.array(_all)


    t2 = time.time()
    print("Distance calc took ", t2 - t1, " seconds")
    return _all


  def show_im(self, i):
    #print(self.X[:,i])
    full_fname = self.fnames[i]
    img = imread(full_fname)
    img = resize(img, IMSIZE, anti_aliasing=True)
    plt.imshow(img)
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
      plt.figure(figsize=(16,12))
      thismanager = get_current_fig_manager()
      thismanager.window.wm_geometry("+50+50")

    for i in range(N):
      plt.subplot(QUERIES_PER_PLOT, N, (cnt-1)*N + i+1)
      self.show_im(closest[i])
      if i == 0:
          plt.title(self.qidx+1)

      # Show anti-examples
      # plt.subplot(2,N, N + i+1)
      # self.show_im(closest[-(i+1)])

    plt.subplots_adjust(wspace=0.05, hspace=0)

    if cnt == QUERIES_PER_PLOT:
      # plt.show()
      #plt.savefig(time.strftime("%Y%m%d-%H%M%S"))
      plt.savefig('./query/query ' + str(self.qidx-QUERIES_PER_PLOT+2) + '-' + str(self.qidx+1), bbox_inches='tight')



# fname = "./imnet-100/color_hist-100.p"
# fname = "./imnet-val/color_hist-100.p"
# fname = "./imnet-val/color_hist-1000.p"
# fname = "./imnet-val/color_hist-5000.p"
# fname = "./imnet-val/color_hist-10000.p"
# fname = "./imnet-val/color_hist-20000.p"
# fname = "./imnet-val/color_hist-50000.p"

# fname = "./imnet-val/hog-1000.p"
# fname = "./imnet-val/hog-5000.p"
fname = "./imnet-val/hog-50000.p"

t1 = time.time()
print("Loadiong pickle...")
with open(fname, 'rb') as f:
  data = pickle.load(f)
t2 = time.time()
print("Complete in ", t2 - t1, " seconds")

fnames = data['fnames']
X = data['all_vecs'].T
# X = X/np.linalg.norm(X,keepdims=True,axis=0)

nn = NN(X, fnames)
#for i in np.random.permutation(len(fnames)):
for i in range(len(fnames)):
  nn.query_idx(i)




