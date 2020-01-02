import argparse
import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def main():

    chessboard = cv2.imread('circleboard10x5.png',1)
    #ch = 1920
    #cw = 1080
    #chessboard = cv2.resize(chessboard,(ch,cw))
    print chessboard.shape[:2]

    ret2, circles = cv2.findCirclesGrid(chessboard, (10,5), flags = cv2.CALIB_CB_SYMMETRIC_GRID)

    if ret2 == True:
        objpoints.append(objp)
        imgpoints.append(circles)
        cv2.drawChessboardCorners(chessboard, (10, 5), circles, ret2)
        print circles[0][0]
        print circles[8][0]
        print circles[49][0]
        print circles[41][0]
        

    cv2.imshow('img', chessboard)
    #cv2.imshow('inv_img',invgray)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
