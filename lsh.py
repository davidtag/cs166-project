import numpy as np
import glob
from scipy import misc
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pdb
import pickle
from skimage.feature import hog
from skimage.transform import resize

IMSIZE = (224, 224)
IMAGE_DIR = "./imnet-100"

def pickle_write_concat_file(fname, im_fnames, all_vecs):
  concat_data = {"fnames":im_fnames, "all_vecs": all_vecs}
  with open(fname + ".p", 'wb') as f:
    pickle.dump(concat_data, f)

def hog_get(image):
  fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, multichannel=True)
  # fd, hog_image = hog(image)
  # print(fd.shape)
  # plt.subplot(211)
  # plt.imshow(hog_image, cmap=plt.cm.gray)
  # plt.subplot(212)
  # plt.imshow(image)
  # plt.show()
  # pdb.set_trace()
  return fd

def per_color_avg(im):
  return im.mean(axis=0).mean(axis=0)

def im2vectors(im_fnames):
  for fname in im_fnames:
    im = misc.imread(fname)
    im = resize(im, IMSIZE, anti_aliasing=True)

    # plt.imshow(im)
    # plt.show()
    data = {}
    data['per_color_avg'] = per_color_avg(im)
    data['hog'] = hog_get(im)
    hog_length = len(data['hog'])
    imvec_fname = fname + ".p"
    with open(imvec_fname, 'wb') as f:
      pickle.dump(data, f)

def concat_vectors(im_fnames):
  vec_fnames = glob.glob(IMAGE_DIR + "/*.p")
  for i, fname in enumerate(vec_fnames):
    with open(fname, 'rb') as f:
      curr = pickle.load(f)

    if i == 0:
      pc_avgs = np.zeros((len(vec_fnames), len(curr['per_color_avg'])))
      hogs = np.zeros((len(vec_fnames), len(curr['hog'])))

    pc_avgs[i,:] = curr['per_color_avg']
    hogs[i,:] = curr['hog']

  pickle_write_concat_file(IMAGE_DIR + "_pc_avg", im_fnames, pc_avgs)
  pickle_write_concat_file(IMAGE_DIR + "_hog", im_fnames, hogs)

########## Test load summary data ############
im_fnames = glob.glob(IMAGE_DIR + "/*.JPEG")
im2vectors(im_fnames);
concat_vectors(im_fnames);
concat_fname = IMAGE_DIR + "_hog.p"
with open(concat_fname, 'rb') as f:
  data = pickle.load(f)
pdb.set_trace()

