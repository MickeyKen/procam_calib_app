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


### termination criteria ###
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


### prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) ###
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)


### Arrays to store object points and image points from all the images. ###
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpeg')


### for proj matrix ###
objectPoints = []
projCirclePoints = []


### for ordinal calibration ###
objpoints_c = [] # 3d point in real world space
imgpoints_c = [] # 2d points in image plane.

w_proj = 1024
h_proj = 768

K = np.array([[ 1.0464088296606685e+03, 0., 9.6962285013582118e+02],
              [0. , 1.0473601981442353e+03, 5.3418043955010319e+02],
              [0., 0., 1. ]],np.float32)

d = np.array([[ 4.3977277514639868e-02, -6.2933078892199332e-02,
       -5.7377837329916246e-04, 7.8218303056817190e-04,
       1.4687866930870116e-02 ]],np.float32)

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

#########################################
### get circle 1024*768 for projector ###
#########################################
args = get_args()
w = args.width
h = args.height
margin = args.margin_size
block_size = args.block_size
radius = args.radius
chessboard = np.ones((block_size * h + margin * 2, block_size * w + margin * 2), dtype=np.uint8) * 255

for y in range(h):
  for x in range(w):
    cx = int((x + 0.5) * block_size + margin)
    cy = int((y + 0.5) * block_size + margin)
    cv.circle(chessboard, (cx, cy), radius, 0, thickness=-1)

chessboard = cv.resize(chessboard,(w_proj,h_proj))

ret_c, circles_c = cv.findCirclesGrid(chessboard, (10,7), flags = cv.CALIB_CB_SYMMETRIC_GRID)

if ret_c == True:
  objpoints_c.append(objp)
  imgpoints_c.append(circles_c)
  cv.drawChessboardCorners(chessboard, (10, 7), circles_c, ret_c)


img_num = 1

########################
### main calibration ###
########################
for fname in images:

  img = cv.imread(fname)
  new_img = img.copy()

  ###  for chessboard  ###
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  ###  for circle grid  ###
  orgHeight, orgWidth = img.shape[:2]
  for i in range(orgHeight):
    for j in range(orgWidth):
      b = img[i][j][0]
      g = img[i][j][1]
      r = img[i][j][2]
      if b > 230 and g > 230 and r > 230:
        img[i][j][0] = 0
        img[i][j][1] = 0
        img[i][j][2] = 0
  gray_for_circle = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


  ### Find the chess and circle board corners  ###
  ret, corners = cv.findChessboardCorners(gray, (9,7), None)
  ret2, circles = cv.findCirclesGrid(gray_for_circle, (10,7), flags = cv.CALIB_CB_SYMMETRIC_GRID)


  ### If found, add object points, image points (after refining them) ###
  if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (9,9), (-1,-1), criteria)
    cv.drawChessboardCorners(new_img, (9,7), corners2, ret)

    ret, rvec, tvec = cv.solvePnP(pW, corners2, K, d)
    rmat = cv.Rodrigues(rvec)[0]
    #rmat [0][0] ~ [2][2]
    rmat[0][2] = 0.020
    rmat[1][2] = 0.200
    rmat[2][2] = 0.956
    # print rmat
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
    #print h


  ### If gound, add objects point ###
  if ret2 == True:
    #print circles[0][0]
    #print circles[9][0]
    #print circles[69][0]
    #print circles[60][0]
    cv.drawChessboardCorners(new_img, (10, 7), circles, ret2)
    proj_pW = np.zeros([7 * 10, 3], dtype=np.float32)

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
    objectPoints.append(proj_pW)
    projCirclePoints.append(circles_c)

  #cv.imshow('img', new_img)
  #cv.waitKey(1000)
  #cv.destroyAllWindows()
  # cv.imwrite( "output_" + str(img_num) + ".jpg", new_img )
  img_num += 1

ret, K_proj, dist_coef_proj, rvecs, tvecs = cv.calibrateCamera(objectPoints,
                                                                projCirclePoints,
                                                               (w_proj, h_proj),
                                                               None,
                                                               None)
                                                    # flags = cv.CALIB_USE_INTRINSIC_GUESS
print("proj calib mat after\n%s"%K_proj)
print("proj dist_coef %s"%dist_coef_proj.T)
print("calibration reproj err %s"%ret)
