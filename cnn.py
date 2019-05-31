import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pdb
import time
import glob
import os

def filedir2batch(img_dir, NMAX):
  img_pattern = os.path.join(img_dir, '*.JPEG')
  img_fnames = sorted(glob.glob(img_pattern))
  if NMAX > 0:
    img_fnames = img_fnames[:NMAX]

  image_batch = np.zeros((NMAX, 224, 224, 3), dtype=np.uint8)
  for i, img_fname in enumerate(img_fnames):
    original = load_img(img_fname, target_size=(224, 224))
    image_batch[i,:,:,:] = img_to_array(original)
  return image_batch

class cnn:
  def __init__(self, name):

    if name == "mobilenet":
      model_loader = mobilenet.MobileNet
      self.img_preprocessor =  mobilenet.preprocess_input
    elif name == "vgg16":
      model_loader = vgg16.VGG16
      self.img_preprocessor =  vgg16.preprocess_input
    elif name == "inception_v3":
      model_loader = inception_v3.InceptionV3
      self.img_preprocessor =  inception_v3.preprocess_input
    elif name == "resnet50":
      model_loader = resnet50.ResNet50
      self.img_preprocessor =  resnet50.preprocess_input
    else:
      raise Exception("unrecognized model name: " + name)

    print("loading model " +  name)
    t1 = time.time()
    self.model = model_loader(weights='imagenet', include_top=False, pooling='avg')
    t2 = time.time()
    print("load time: ", t2 - t1)

  def predict(self, image_batch):
    if image_batch.ndim == 3:
      image_batch = np.expand_dims(image_batch, axis=0)
    processed_images = self.img_preprocessor(image_batch.copy())
    features = self.model.predict(processed_images)
    return features

if __name__== "__main__":
  models = []
  models.append(cnn('vgg16'))
  models.append(cnn('inception_v3'))
  models.append(cnn('resnet50'))
  models.append(cnn('mobilenet'))
   
  filedir = 'imnet-val/imgs/'
  for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for j in range(3):
      t1 = time.time()
      image_batch = filedir2batch(filedir, i)
      for model in models:
        model.predict(image_batch)
      t2 = time.time()
      print(i, t2-t1, (t2-t1)/i)
