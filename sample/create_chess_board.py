import numpy as np
import cv2 as cv
import glob
import argparse
import matplotlib
import matplotlib.pyplot as plt

############################
### create chessboard pW ###
############################
row = 9
col = 7
corner_num = (row, col)
size = 2.5

pW = np.empty([row *col, 3], dtype=np.float32)

for i_row in range(0, row):
  for i_col in range(0, col):
    pW[i_row * col + i_col] = np.array([size * i_col, size * i_row, 0], dtype = np.float32)

# print pW
