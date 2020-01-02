import numpy as np
import cv2 as cv
import glob
import argparse
import matplotlib
import matplotlib.pyplot as plt

############################################################################################################
### for get .jpeg file ###
def get_args():
    parser = argparse.ArgumentParser(description="This script creates a circleboard image for calibration")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--margin_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=100)
    parser.add_argument("--radius", type=int, default=30)
    args = parser.parse_args()
    return args
args = get_args()
### projector parameter ###
proj_width = 1920
proj_height = 1080

### chessboard parameter ###
row = 9
col = 7
corner_num = (row, col)
size = 2.5

### camera parameter ###
K = np.array([[ 1.0464088296606685e+03, 0., 9.6962285013582118e+02],
              [0. , 1.0473601981442353e+03, 5.3418043955010319e+02],
              [0., 0., 1. ]],np.float32)

d = np.array([[ 4.3977277514639868e-02, -6.2933078892199332e-02,
       -5.7377837329916246e-04, 7.8218303056817190e-04,
       1.4687866930870116e-02 ]],np.float32)

### termination criteria ###
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


### prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) ###
objp = np.zeros((args.width*args.height,3), np.float32)
objp[:,:2] = np.mgrid[0:args.width,0:args.height].T.reshape(-1,2)


### Arrays to store object points and image points from all the images. ###
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

### for ordinal calibration ###
objpoints_c = [] # 3d point in real world space
imgpoints_c = [] # 2d points in image plane.

### for projector matrix ###
objectPoints = []
projCirclePoints = []
############################################################################################################


############################
### create chessboard pW ###
############################
def chessboard_pW():

    pW = np.empty([row *col, 3], dtype=np.float32)

    for i_row in range(0, row):
      for i_col in range(0, col):
        pW[i_row * col + i_col] = np.array([size * i_col, size * i_row, 0], dtype = np.float32)

    return pW

def create_gray_img(in_img):
    ###  for chessboard  ###
    gray = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)

    ###  for circle grid  ###
    orgHeight, orgWidth = in_img.shape[:2]
    for i in range(orgHeight):
        for j in range(orgWidth):
          b = in_img[i][j][0]
          g = in_img[i][j][1]
          r = in_img[i][j][2]
          if b > 230 and g > 230 and r > 230:
            in_img[i][j][0] = 0
            in_img[i][j][1] = 0
            in_img[i][j][2] = 0
    gray_for_circle = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)

    return gray, gray_for_circle


##########################################
### get circle 1920*1080 for projector ###
#########################################

chessboard = cv.imread('circleboard10x5.png',1)

ret_c, circles_c = cv.findCirclesGrid(chessboard, (args.width,args.height), flags = cv.CALIB_CB_SYMMETRIC_GRID)

if ret_c == True:
  objpoints_c.append(objp)
  imgpoints_c.append(circles_c)
  cv.drawChessboardCorners(chessboard, (args.width, args.height), circles_c, ret_c)

#####################
### main function ###
#####################
img_num = 1
images = glob.glob('*.jpeg')

for fname in images:

  img = cv.imread(fname)
  copy_img = img.copy()
  gray, gray_for_circle = create_gray_img(copy_img)
  cv.imshow("circleboard", gray_for_circle)
  cv.waitKey(0)
  cv.destroyAllWindows()
