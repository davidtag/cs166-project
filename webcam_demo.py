import numpy as np
import cv2
import time
import cnn
import pdb
import linear_scan
import featurize
from skimage.transform import resize
import os
from lsh import *
from utils import *

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

IMSIZE = (224, 224)
aspect = 16/9
WIDTH = 300
HEIGHT = int(WIDTH*aspect)
DISPLAY_SIZE =(HEIGHT, WIDTH)
# fname1 = "./imnet-val/cnn-50000.p"
# nn = linear_scan.NN(fname1)


PATH_IMGS     = "imnet-val/val/"
FILE_FEATURES = "imnet-val/cnn-50000.p"
FNAME_OFFSET  = 48 #prefix of stored file names to chop off
IMSIZE        = (224, 224)

data = dataset(FILE_FEATURES,PATH_IMGS,IMSIZE,normalize=True,fname_offt=FNAME_OFFSET)
X = data.X
d,n = X.shape

b = 200 #hash bits
M = 20  #number of permutations
k = 3
L = 4
model = LSH(X=X,b=b,M=M)

cnn_model = cnn.cnn("mobilenet")

cap = cv2.VideoCapture(0)
time.sleep(0.1)
pause = False

empty = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)

while(True):
  if not pause:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 0)

    # frame = image_resize(frame, width = DISPLAY_SIZE)
    resized = cv2.resize(frame, DISPLAY_SIZE, interpolation = cv2.INTER_AREA)
    row1 = np.hstack((resized, empty))
    row1 = np.hstack((row1, empty))
    row2 = np.hstack((empty, empty))
    row2 = np.hstack((row2, empty))
    display = np.vstack((row1, row2))

    # Display the resulting frame
    cv2.imshow('frame',display)

  x = cv2.waitKey(1)
  # print(x)
  if x == ord('q'):
    break
  elif x == 32: # space bar
    # pause
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    reshaped_img = resize(img, IMSIZE, anti_aliasing=True)

    t1 = time.time()
    q = cnn_model.predict(reshaped_img*255).flatten()
    q = np.expand_dims(q, axis=1)
    print("cnn time {:.3f}".format(time.time()-t1))

    t1 = time.time()
    # closest = nn.query(q, reshaped_img, False )
    rank_3_lsh_ip   = model.approx_top_k(q,k,L,refine="innerprod")
    print("lsh query time {:.3f}".format(time.time()-t1))

    imgs = []
    for i in rank_3_lsh_ip:
      fname = data.get_img_path(i)
      img = cv2.imread(fname)
      img = cv2.resize(img, DISPLAY_SIZE, interpolation = cv2.INTER_AREA)
      imgs.append(img)

    t1 = time.time()
    exact   = model.exact_top_k(q,k)
    print("exact query time {:.2f}".format(time.time()-t1))
    imgs_exact = []
    for i in exact:
      fname = data.get_img_path(i)
      img = cv2.imread(fname)
      img = cv2.resize(img, DISPLAY_SIZE, interpolation = cv2.INTER_AREA)
      imgs_exact.append(img)


    row1 = np.hstack((resized, imgs[0]))
    row1 = np.hstack((row1, imgs[1]))
    # row2 = np.hstack((imgs[1], imgs[2]))
    row2 = np.hstack((empty, imgs_exact[0]))
    row2 = np.hstack((row2, imgs_exact[1]))
    display = np.vstack((row1, row2))
    cv2.imshow('frame', display)
    y = cv2.waitKey(0)
    if y == ord('q'):
      break

    # pause = not pause

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()