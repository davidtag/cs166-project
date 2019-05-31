import numpy as np
import glob
from imageio import imread
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pdb
import pickle
from skimage.feature import hog, daisy, orb
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
import time
import os
import shutil
import cnn
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

IMSIZE = (224, 224)
# BASE_DIR = "./imnet-100"
# BASE_DIR = "./imnet-val"
# BASE_DIR = "/Volumes/oddish1tb/cs166-project/imnet-val"
BASE_DIR = "/Volumes/oddish1tb/cs166-project/imnet-test"
# BASE_DIR = "./imnet-test"
NMAX = -1

def mkdir_safe(path):
  if not os.path.exists(path):
    os.mkdir(path)

def mkdir_clean(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.mkdir(path)

def feature_fname_get(feature_name):
  img_fnames = get_img_fnames()
  N = len(img_fnames)
  return os.path.join(BASE_DIR, feature_name + "-" + str(len(img_fnames)) + ".p")

def pickle_write_concat_file(base_dir, feature_name, img_fnames, all_vecs):
  concat_data = {"fnames":img_fnames, "all_vecs": all_vecs}
  fname = feature_fname_get(feature_name)
  with open(fname, 'wb') as f:
    pickle.dump(concat_data, f, pickle.HIGHEST_PROTOCOL)

def plot_features(image, feature_image):
  plt.subplot(211)
  plt.imshow(feature_image)
  plt.subplot(212)
  plt.imshow(image)
  plt.show()

def hog_get(img):
  fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

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
  bin_edges = np.tile(bin_edges, (img.shape[2],1))

  flat_img = img.reshape((-1, img.shape[2] )) # lose spatial dimension
  H, edges = np.histogramdd(flat_img, bins = bin_edges)
  return H.flatten()

def cnn_get(img, cnn_model):
  features = cnn_model.predict(np.expand_dims(img, axis=0))
  return features[0,:]

def per_color_avg(im):
  return im.mean(axis=0).mean(axis=0)

def first_pixel(im):
  return im[0,0,:]

def img_dir_from_base_dir(base_dir):
  return os.path.join(base_dir, "imgs")

def raw_dir_from_base_dir(base_dir):
  return os.path.join(base_dir, "imgs_raw_npy")

def features_dir_from_base_dir(base_dir):
  return os.path.join(base_dir, "feature_vectors")

def raw_name_from_img_name(img_fname):
  path, file = os.path.split(img_fname)
  file_no_ext, ext = os.path.splitext(file)
  base_dir, _ = os.path.split(path)
  raw_dir = raw_dir_from_base_dir(base_dir)
  return os.path.join(raw_dir, file_no_ext + '.npy')

def features_name_from_img_name(img_fname):
  path, file = os.path.split(img_fname)
  file_no_ext, ext = os.path.splitext(file)
  base_dir, _ = os.path.split(path)
  features_dir = features_dir_from_base_dir(base_dir)
  return os.path.join(features_dir, file_no_ext + '.p')

def get_img_fnames():
  img_dir = img_dir_from_base_dir(BASE_DIR)
  img_pattern = os.path.join(img_dir, '*.JPEG')
  img_fnames = sorted(glob.glob(img_pattern))
  if NMAX > 0:
    img_fnames = img_fnames[:NMAX]
  return img_fnames

def standardize_img(img):
  # img = resize(img, IMSIZE, anti_aliasing=True)
  if (img.ndim == 2):
    img = gray2rgb(img)

  if img.shape[2] != 3:
    pdb.set_trace()

  return img

def print_timing(t0, avg_elapsed, msg, N, i):
  t1 = time.time()
  elapsed = t1 - t0
  avg_elapsed += 0.01*(elapsed - avg_elapsed)
  remaining = (N - i)*avg_elapsed
  print(msg + " iter: {}/{}, curr: {:5.2f} sec, avg: {:5.2f} sec, remaining: {:6.1f} minutes".format(i, N, elapsed, avg_elapsed, remaining/60.))
  return avg_elapsed

def imgs2npy():
  img_fnames = get_img_fnames()
  N = len(img_fnames)

  raw_dir = raw_dir_from_base_dir(BASE_DIR)
  # mkdir_clean(raw_dir)
  mkdir_safe(raw_dir)

  avg_elapsed = 0
  for i, img_fname in enumerate(img_fnames):
    t0 = time.time()

    raw_fname = raw_name_from_img_name(img_fname)
    if os.path.isfile(raw_fname):
      continue
    img = img_to_array(load_img(img_fname), dtype='uint8')
    np.save(raw_fname, img, allow_pickle=False)

    avg_elapsed = print_timing(t0, avg_elapsed, "imgs2npy", N, i)

def npy2features():
  img_fnames = get_img_fnames()
  N = len(img_fnames)

  cnn_model = cnn.cnn("mobilenet")

  features_dir = features_dir_from_base_dir(BASE_DIR)
  # mkdir_clean(features_dir)
  mkdir_safe(features_dir)

  avg_elapsed = 0
  for i, img_fname in enumerate(img_fnames):
    t0 = time.time()

    raw_fname = raw_name_from_img_name(img_fname)
    features_fname = features_name_from_img_name(img_fname)
    if os.path.isfile(features_fname):
      continue
    img = np.load(raw_fname)
    img = standardize_img(img)
    reshaped_img = resize(img, IMSIZE, anti_aliasing=True)
    # plt.imshow(im)
    # plt.show()

    try: 
      data = {}
      # data['per_color_avg'] = per_color_avg(img)
      # data['hog'] = hog_get(reshaped_img)
      # data['daisy'] = daisy_get(img)
      data['color_hist'] = color_hist(reshaped_img)
      # data['first_pixel'] = first_pixel(img)
      data['cnn'] = cnn_get(img, cnn_model)
      with open(features_fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
      print("Failed to featurize image " + img_fname)
      print(e)


    avg_elapsed = print_timing(t0, avg_elapsed, "npy2features", N, i)

def concat_features():
  img_fnames = get_img_fnames()
  N = len(img_fnames)

  concatenated = {}
  avg_elapsed = 0
  for i, img_fname in enumerate(img_fnames):
    t0 = time.time()

    features_fname = features_name_from_img_name(img_fname)

    with open(features_fname, 'rb') as f:
      curr = pickle.load(f)

    # Initialize matrices
    if i == 0:
      for feature_name, value in curr.items():
        concatenated[feature_name] = np.zeros((N, len(value)))

    # Copy data into row
    for feature_name, value in curr.items():
      concatenated[feature_name][i,:] = value

    avg_elapsed = print_timing(t0, avg_elapsed, "concat", N, i)

  # Write concatenated data to file
  print('Concatenating finished, saving to file...')
  for feature_name, value in concatenated.items():
    pickle_write_concat_file(BASE_DIR, feature_name, img_fnames, value)


if __name__ == '__main__':

  ########## Run Main ############
  t1 = time.time()
  imgs2npy()
  npy2features();
  concat_features();
  t2 = time.time()
  print(t2-t1)

  ########## Test load summary data ############
  img_fnames = get_img_fnames()
  N = len(img_fnames)
  concat_fname = feature_fname_get('cnn')
  with open(concat_fname, 'rb') as f:
    data = pickle.load(f)

