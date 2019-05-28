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

def file2batch(filename):
  original = load_img(filename, target_size=(224, 224))
  numpy_image = img_to_array(original)
  image_batch = np.expand_dims(numpy_image, axis=0)
  return image_batch

def predict(model, image_batch, preprocessor):
  t1 = time.time()
  processed_image = preprocessor(image_batch.copy())
  features = model.predict(processed_image).flatten()
  t2 = time.time()
  print(features.shape, t2 - t1)
  return features

def get_model(model_loader):
  return model_loader(weights='imagenet', include_top=False, pooling='avg')

vgg_model = get_model(vgg16.VGG16)
inception_model = get_model(inception_v3.InceptionV3)
resnet_model = get_model(resnet50.ResNet50)
mobilenet_model = get_model(mobilenet.MobileNet)
 
filename = 'imnet-100/imgs/ILSVRC2010_val_00000001.JPEG'
image_batch = file2batch(filename)
f1 = predict(vgg_model, image_batch, vgg16.preprocess_input) 
f2 = predict(inception_model, image_batch, inception_v3.preprocess_input) 
f3 = predict(resnet_model, image_batch, resnet50.preprocess_input) 
f4 = predict(mobilenet_model, image_batch, mobilenet.preprocess_input) 
