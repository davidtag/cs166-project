import numpy as np
import glob
from scipy import misc
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pdb
import pickle
from skimage.feature import hog, daisy, orb
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
import time

IMSIZE = (224, 224)
# IMAGE_DIR = "./imnet-100"
IMAGE_DIR = "./imnet-val"

def pickle_write_concat_file(fname, im_fnames, all_vecs):
  concat_data = {"fnames":im_fnames, "all_vecs": all_vecs}
  with open(fname + ".p", 'wb') as f:
    pickle.dump(concat_data, f)

def plot_features(image, feature_image):
  plt.subplot(211)
  plt.imshow(feature_image)
  plt.subplot(212)
  plt.imshow(image)
  plt.show()

def hog_get(img):
  try:
    fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
  except:
    pdb.set_trace()
  # fd, hog_img = hog(img, visualize=True, multichannel=True)
  # print(fd.shape)
  # plot_features(img, hog_img)
  return fd

def daisy_get(img):
  gray = rgb2gray(img)
  descs, descs_img = daisy(gray, step=80, radius=28, rings=2, histograms=6, orientations=8, visualize=True)
  descs = descs.flatten()
  # plot_features(img, descs_img)
  return descs

def color_hist(img):
  nbins = 4
  bin_edges = np.linspace(0, 1, nbins+1)
  bin_edges = np.tile(bin_edges, (3,1))

  flat_img = img.reshape((-1,3)) # lose spatial dimension
  H, edges = np.histogramdd(flat_img, bins = bin_edges)
  return H.flatten()

def per_color_avg(im):
  return im.mean(axis=0).mean(axis=0)

def im2vectors(im_fnames):
  for i, fname in enumerate(im_fnames):
    t0 = time.time()
    print(i, len(im_fnames))
    img = misc.imread(fname)
    img = resize(img, IMSIZE, anti_aliasing=True)
    if (img.ndim == 2):
      img = gray2rgb(img)


    # plt.imshow(im)
    # plt.show()
    data = {}
    t1 = time.time()
    # data['per_color_avg'] = per_color_avg(img)
    t2 = time.time()
    data['hog'] = hog_get(img)
    t3 = time.time()
    # data['daisy'] = daisy_get(img)
    t4 = time.time()
    data['color_hist'] = color_hist(img)
    t5 = time.time()
    # print(t2-t1, t3-t2, t4-t3, t5-t4)
    vec_fname = fname + ".p"
    with open(vec_fname, 'wb') as f:
      pickle.dump(data, f)
    t6 = time.time()
    print(t6 - t0)

def concat_vectors(im_fnames):
  vec_fnames = glob.glob(IMAGE_DIR + "/*.p")
  for i, fname in enumerate(vec_fnames):
    with open(fname, 'rb') as f:
      curr = pickle.load(f)

    if i == 0:
      # pc_avgs = np.zeros((len(vec_fnames), len(curr['per_color_avg'])))
      hogs = np.zeros((len(vec_fnames), len(curr['hog'])))
      # daisy = np.zeros((len(vec_fnames), len(curr['daisy'])))
      color_hist = np.zeros((len(vec_fnames), len(curr['color_hist'])))

    # pc_avgs[i,:] = curr['per_color_avg']
    hogs[i,:] = curr['hog']
    # daisy[i,:] = curr['daisy']
    color_hist[i,:] = curr['color_hist']

  # pickle_write_concat_file(IMAGE_DIR + "_pc_avg", im_fnames, pc_avgs)
  pickle_write_concat_file(IMAGE_DIR + "_hog", im_fnames, hogs)
  # pickle_write_concat_file(IMAGE_DIR + "_daisy", im_fnames, daisy)
  pickle_write_concat_file(IMAGE_DIR + "_color_hist", im_fnames, color_hist)

########## Test load summary data ############
im_fnames = glob.glob(IMAGE_DIR + "/*.JPEG")
im2vectors(im_fnames);
concat_vectors(im_fnames);
concat_fname = IMAGE_DIR + "_color_hist.p"
with open(concat_fname, 'rb') as f:
  data = pickle.load(f)

