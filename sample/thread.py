import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt

# read an image
img = cv2.imread('camera_imagedata_2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
invgray = cv2.bitwise_not(gray)
 
# show image format (basically a 3-d array of pixel color info, in BGR format)
#print(img)

# convert image to RGB color for matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# show image with matplotlib
plt.imshow(img)

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
# grayscale image represented as a 2-d array
#print(gray_img)

_, threshold_img = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)
 
# show image
threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
plt.imshow(threshold_img)
plt.show()
