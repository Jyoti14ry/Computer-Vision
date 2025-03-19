//FEATURE IDENTIFICATION IN FINGERPRINT RECOGNITION SYSTEM

!pip install opencv-python
!pip install opencv-python-headless
!pip install opencv-contrib-python
from os import path
if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
    !wget https://biolab.csr.unibo.it/samples/fr/files.zip
    !unzip files.zip
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
from ipywidgets import interact
fingerprint = cv.imread('samples/sample_1_1.png', cv.IMREAD_GRAYSCALE)
show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')
gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
show((gx, 'Gx'), (gy, 'Gy'))
gx2, gy2 = gx*2, gy*2
gm = np.sqrt(gx2 + gy2)
show((gx2, 'Gx*2'), (gy2, 'Gy*2'), (gm, 'Gradient magnitude'))
sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
show(sum_gm, 'Integral of the gradient magnitude')
thr = sum_gm.max() * 0.2
mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
show(fingerprint, mask, cv.merge((mask, fingerprint, fingerprint)))
W = (23, 23)
gxx = cv.boxFilter(gx2, -1, W, normalize = False)
gyy = cv.boxFilter(gy2, -1, W, normalize = False)
gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
gxx_gyy = gxx - gyy
gxy2 = 2 * gxy
orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
sum_gxx_gyy = gxx + gyy
strengths = np.divide(cv.sqrt((gxx_gyy*2 + gxy2*2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)
show(draw_orientations(fingerprint, orientations, strengths, mask, 1, 16), 'Orientation image')
