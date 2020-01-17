import numpy as np
import cv2

img = cv2.imread('board/circleboard10x5.png',cv2.IMREAD_COLOR)
#orgHeight, orgWidth = img.shape[:2]
#size = (orgHeight/2, orgWidth/2)
#img = cv2.resize(img , (int(1920), int(1080)))

cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('screen',img)


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
