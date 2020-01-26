# coding: utf-8
import numpy, cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
patw, path =  6, 4
objp = numpy.zeros((patw*path, 3))
for i in range(patw*path):
    objp[i,:2] = numpy.array([i % patw, i / patw], numpy.float32)
objp_list, imgp_list = [], []
while 1:
    stat, image = cap.read(1)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (patw, path),None)
    cv2.drawChessboardCorners(image, (patw, path), corners, ret)
    cv2.imshow('Camera', image)
    key = cv2.waitKey(10)
    if key == 0x1b: # ESC
        break
    elif key == 0x20 and ret == True:
        #print corners
        print 'Saved!'
        objp_list.append(objp.astype(numpy.float32))
        imgp_list.append(corners)
if len(objp_list) >= 3:
    K = numpy.zeros((3,3), float)
    dist = numpy.zeros((5,1), float)
    cv2.calibrateCamera(objp_list, imgp_list, (image.shape[1], image.shape[2]), K, dist)
    print 'K = ¥n', K
    numpy.savetxt('K.txt', K)
    print 'Dist coeff = ¥n', dist
    numpy.savetxt('distCoef.txt', dist)
cap.release()
cv2.destroyAllWindows()
