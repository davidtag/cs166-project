import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import time
import cnn
import pdb
import linear_scan
import featurize
from skimage.transform import resize
import os
from imageio import imread

root = tk.Tk()
root.withdraw()



IMSIZE = (224, 224)
# DISPLAY_SIZE = 600
fname = "./imnet-val/cnn-50000.p"
# fname = "./imnet-val/color_hist-50000.p"
nn = linear_scan.NN(fname)

cnn_model = cnn.cnn("mobilenet")


while True:
  # file_path = filedialog.askopenfilename()
  file_path = "~/Pictures/Lightroom CC Saved Photos/Ngoro 2 cut 2/_MG_0925.jpg"
  img = imread(file_path)
  reshaped_img = resize(img, IMSIZE, anti_aliasing=True)
  # q = featurize.color_hist(reshaped_img)

  q1 = cnn_model.predict(reshaped_img*255).flatten()
  # q2 = cnn_model.predict(img).flatten()
  # pdb.set_trace()
  closest = nn.query(q1, reshaped_img, doplot=True )
