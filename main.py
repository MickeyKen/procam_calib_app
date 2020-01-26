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
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--margin_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=100)
    parser.add_argument("--radius", type=int, default=20)
    args = parser.parse_args()
    return args
args = get_args()
### projector parameter ###
proj_width = 1920
proj_height = 1080

### chessboard parameter ###
row = 6
col = 4
corner_num = (row, col)
size = 3

### camera parameter ###
K = np.array([[ 1.71043410e+03, 0.000000, 1.07773094e+03],
              [0.000000, 1.69437227e+03, 3.52642480e+02],
              [0., 0., 1. ]],np.float32)

d = np.array([[ 0.06935322, -0.12890731, -0.01425927, 0.0161514, 0.05991636]],np.float32)

### termination criteria ###
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


### prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) ###
objp = np.zeros((args.width*args.height,3), np.float32)
objp[:,:2] = np.mgrid[0:args.width,0:args.height].T.reshape(-1,2)


### Arrays to store object points and image points from all the images. ###
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


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


##########################################
### get circle 1920*1080 for projector ###
#########################################
def get_circle_grid():
    chessboard = cv.imread('board/circleboard10x5.png',1)
    chessboard = cv.bitwise_not(chessboard)

    ret_circle, circles_circle = cv.findCirclesGrid(chessboard, (args.width,args.height), flags = cv.CALIB_CB_SYMMETRIC_GRID)

    if ret_circle == True:
      cv.drawChessboardCorners(chessboard, (args.width, args.height), circles_circle, ret_circle)

    return circles_circle

#####################
### main function ###
#####################

img_num = 1
images = glob.glob('data/*.jpg')

for fname in images:

  img = cv.imread(fname)
  # print fname
  img2 = cv.imread("data2/" + fname.split('/')[1])
  # print "data2/" + fname.split('/')[1]
  # print "*** "

  copy_img = img.copy()
  copy_img2 = img2.copy()

  gray = cv.cvtColor(copy_img, cv.COLOR_BGR2GRAY)
  gray_for_circle = cv.cvtColor(copy_img2, cv.COLOR_BGR2GRAY)


  ### Find the chess and circle board corners  ###
  ret, corners = cv.findChessboardCorners(gray, (6,4), None)
  ret2, circles = cv.findCirclesGrid(gray_for_circle, (10,5), flags = cv.CALIB_CB_SYMMETRIC_GRID)

  if ret == True and ret2 == True:
      #print ("success")
      ### main calibration ###
      corners2 = cv.cornerSubPix(gray, corners, (9,9), (-1,-1), criteria)
      cv.drawChessboardCorners(img, (6,4), corners2, ret)

      pW = chessboard_pW()

      ret, rvec, tvec = cv.solvePnP(pW, corners2, K, d)
      rmat = cv.Rodrigues(rvec)[0]
      #rmat [0][0] ~ [2][2]
      rmat[0][2] = 0.020
      rmat[1][2] = 0.200
      rmat[2][2] = 0.956
      #  rmat
      ARt = np.dot(K, rmat)
      ARt = np.linalg.inv(ARt)
      # print H

      copy_pW = pW.copy()

      c = 0
      for i in corners2:
          keisan_mat = np.float32([[0.0,  0.0, 1.0]])
          keisan_mat[0][0] = i[0][0]
          keisan_mat[0][1] = i[0][1]
          keisan_mat = keisan_mat.T
          keisan_mat = np.dot(ARt, keisan_mat)
          keisan_mat = keisan_mat / keisan_mat[2][0]
          copy_pW[c][0] = keisan_mat[0][0]
          copy_pW[c][1] = keisan_mat[1][0]
          # print copy_pW
          c += 1
      # print copy_pW

      h, status = cv.findHomography(copy_pW, corners2)
      h = np.linalg.inv(h)
      cv.drawChessboardCorners(img, (args.width, args.height), circles, ret2)
      proj_pW = np.zeros([args.height * args.width, 3], dtype=np.float32)

      count = 0
      for i in circles:
        mat = np.float32([[0.0,  0.0, 1.0]])
        mat[0][0] = i[0][0]
        mat[0][1] = i[0][1]
        mat = mat.T
        mat = np.dot(h, mat)
        mat = mat / mat[2][0]
        proj_pW[count][0] = mat[0][0]
        proj_pW[count][1] = mat[1][0]
        # print count
        count += 1
      #print proj_pW
      circles_circle = get_circle_grid()
      objectPoints.append(proj_pW)
      projCirclePoints.append(circles_circle)

  img_num += 1
  cv.imshow('screen',img)
  cv.waitKey(0)

# print (objectPoints)
# print (projCirclePoints)

ret, K_proj, dist_coef_proj, rvecs, tvecs = cv.calibrateCamera(objectPoints,
                                                                projCirclePoints,
                                                               (proj_width, proj_height),
                                                               None,
                                                               None,
                                                               None,
                                                               None)
                                                    # flags = cv.CALIB_USE_INTRINSIC_GUESS
print("proj calib mat after\n%s"%K_proj)
print("proj dist_coef %s"%dist_coef_proj.T)
print("calibration reproj err %s"%ret)
