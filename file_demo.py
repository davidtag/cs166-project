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
  file_path = filedialog.askopenfilename()
  img = imread(file_path)
  # reshaped_img = resize(img, IMSIZE, anti_aliasing=True)
  # q = featurize.color_hist(reshaped_img)

  q = cnn_model.predict(img).flatten()

  closest = nn.query(q, img, doplot=True )
