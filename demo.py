import numpy as np
import cv2
import time
import cnn
import pdb
import linear_scan
import featurize
from skimage.transform import resize
import os

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
DISPLAY_SIZE = 600
fname1 = "./imnet-val/cnn-50000.p"
fname2 = "./imnet-val/color_hist-50000.p"
nn1 = linear_scan.NN(fname1)
nn2 = linear_scan.NN(fname2)


cnn_model = cnn.cnn("mobilenet")

cap = cv2.VideoCapture(1)
time.sleep(0.1)
pause = False

while(True):
  if not pause:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = image_resize(frame, width = DISPLAY_SIZE)

    # Display the resulting frame
    cv2.imshow('frame',frame)

  x = cv2.waitKey(1)
  print(x)
  if x == ord('q'):
    break
  elif x == 32: # space bar
    # pause
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    q1 = cnn_model.predict(img).flatten()

    reshaped_img = resize(img, IMSIZE, anti_aliasing=True)
    q2 = featurize.color_hist(reshaped_img)


    closest1 = nn1.query(q1, False )
    closest2 = nn2.query(q2, False )

    for j in range(2):
      for i in range(4):
        if j == 0:
          closest = closest1
          nn = nn1
        else:
          closest = closest2
          nn = nn2

        _, fname = os.path.split(nn.fnames[closest[i]])
        fname = os.path.join("./imnet-val/imgs", fname)

        img = cv2.imread(fname)
        img = image_resize(img, width = DISPLAY_SIZE)
        cv2.imshow('frame', img)
        y = cv2.waitKey(0)
        if y == ord('q'):
          break

    # pause = not pause

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()