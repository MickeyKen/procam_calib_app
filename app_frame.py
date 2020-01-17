import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from app import Ui_MainWindow

import cv2
import numpy as np
# from PIL import Image, ImageDraw, ImageFilter

class Test(QMainWindow):
    def __init__(self,parent=None):
        super(Test, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.save)
        self.ui.pushButton_2.clicked.connect(self.before)
        self.ui.pushButton_3.clicked.connect(self.after)
        self.ui.pushButton_4.clicked.connect(self.calib)
        self.ui.verticalSlider.valueChanged['int'].connect(self.value_change)
        self.ui.verticalSlider_2.valueChanged['int'].connect(self.value_change_2)
        self.ui.verticalSlider_3.valueChanged['int'].connect(self.value_change_3)

        self.b = 0
        self.g = 0
        self.r = 0

        self.img_name = 1

        self.load_img(str(self.img_name), 0,0,0)
        # self.ui.

    def save(self,file_name):
        self.load_img(str(self.img_name), 1,1,0)

    def before(self):
        if self.img_name > 1:
            self.img_name -= 1
        self.load_img(str(self.img_name), 0,0,0)

    def after(self):
        self.img_name += 1
        self.load_img(str(self.img_name), 0,0,0)

    def calib(self):
        self.load_img(str(self.img_name), 1,0,1)

    def value_change(self):
        # self.verticalSlider.value()
        self.ui.lcdNumber.display(self.ui.verticalSlider.value())
        self.b = self.ui.verticalSlider.value()
        self.load_img(str(self.img_name), 1,0,0)

    def value_change_2(self):
        # self.verticalSlider.value()
        self.ui.lcdNumber_2.display(self.ui.verticalSlider_2.value())
        self.g = self.ui.verticalSlider_2.value()
        self.load_img(str(self.img_name), 1,0,0)

    def value_change_3(self):
        # self.verticalSlider.value()
        self.ui.lcdNumber_3.display(self.ui.verticalSlider_3.value())
        self.r = self.ui.verticalSlider_3.value()
        self.load_img(str(self.img_name), 1,0,0)


    def load_img(self, file_name, binary_flag, save_flag, calib_flag):
        # print ("data/" , file_name , ".jpg")
        self.img = cv2.imread("data/" + file_name + ".jpg", cv2.IMREAD_COLOR)
        img = np.copy(self.img)
        # img = np.float32(img)

        if binary_flag == 1:
            orgHeight, orgWidth = img.shape[:2]
            for i in range(orgHeight):
              for j in range(orgWidth):
                b = img[i][j][0]
                g = img[i][j][1]
                r = self.img[i][j][2]
                if b > self.b and g > self.g and r > self.r:
                  img[i][j][0] = 0
                  img[i][j][1] = 0
                  img[i][j][2] = 0

        if save_flag == 1:
            cv2.imwrite("data2/" + file_name + ".jpg", img)

        if calib_flag == 1:
            ret2, circles = cv2.findCirclesGrid(img, (10,5), flags = cv2.CALIB_CB_SYMMETRIC_GRID)
            if ret2 == True:
                cv2.drawChessboardCorners(img, (10, 5), circles, ret2)

        show_img = cv2.resize(img,(480,270))
        self.drawing(show_img)

    def drawing(self,in_img):
        qimg = QImage(in_img, in_img.shape[1], in_img.shape[0], QImage.Format_RGB888)
        # self.uic.imageLabel.setWindowOpacity(0.8)
        self.ui.textBrowser.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Test()
    window.show()
    sys.exit(app.exec_())
