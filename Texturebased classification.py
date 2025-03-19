//TEXTURE-BASED IMAGE CLASSIFICATION USING RECEIVER OPERATING CURVE

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
show(draw_orientations(fingerprint, orientations, strengths, mask, 1, 16), 'Orientation image')
region = fingerprint[10:90,80:130]
show(region)
xs = np.sum(smoothed, 1) # the x-signature of the region
print(xs)

x = np.arange(region.shape[0])
f, axarr = plt.subplots(1,2, sharey = True)
axarr[0].imshow(region,cmap='gray')
axarr[1].plot(xs, x)
axarr[1].set_ylim(region.shape[0]-1,0)
plt.show()
local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
x = np.arange(region.shape[0])
plt.plot(x, xs)
plt.xticks(local_maxima)
plt.grid(True, axis='x')
plt.show()
distances = local_maxima[1:] - local_maxima[:-1]
print(distances)
ridge_period = np.average(distances)
print(ridge_period)
or_count = 8
gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]
nf = 255-fingerprint
all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
show(nf, *all_filtered)
y_coords, x_coords = np.indices(fingerprint.shape)
orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
filtered = all_filtered[orientation_idx, y_coords, x_coords]
enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
show(fingerprint, enhanced)
_, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
show(fingerprint, ridge_lines, cv.merge((ridge_lines, fingerprint, fingerprint)))
skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)
show(skeleton, cv.merge((fingerprint, fingerprint, skeleton)))
def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))
cn_filter = np.array([[  1,  2,  4],  [128,  0, 64],[ 64, 32, 16]])
                      
all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
neighborhood_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
cn = cv.LUT(neighborhood_values, cn_lut)
cn[skeleton==0] = 0
minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]
show(draw_minutiae(fingerprint, minutiae), skeleton, draw_minutiae(skeleton, minutiae))
