import math
import sys
from sys import flags

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from skimage import morphology
import numpy as np
import cv2 as cv
from APPGSI import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.openImage)
        self.pushButton_2.clicked.connect(self.btnGray_Clicked)
        self.pushButton_3.clicked.connect(self.btnBinaryzation_Clicked)
        self.pushButton_4.clicked.connect(self.quzao_Clicked)

    def openImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, '打开图⽚')
        if filename:
            self.captured1 = cv2.imread(str(filename))
            self.captured = cv2.resize(self.captured1, (270, 270), interpolation=cv2.INTER_AREA)
            # OpenCV图像以BGR通道存储，显⽰时需要从BGR转到RGB
            self.captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGray_Clicked(self):
        # 如果没有捕获图⽚，则不执⾏操作
        if not hasattr(self, "captured"):
            return
        self.cpatured = cv2.cvtColor(self.captured, cv2.COLOR_RGB2GRAY)

        rows, columns = self.cpatured.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要⽤Format_Indexed8
        QImg = QImage(self.cpatured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnBinaryzation_Clicked(self):
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return
        self.gray = cv2.cvtColor(self.captured, cv2.COLOR_RGB2GRAY)
        self.binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)

        rows, columns = self.binary.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要⽤Format_Indexed8
        QImg = QImage(self.binary.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_4.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_4.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def quzao_Clicked(self):

        self.binary = 255 - self.binary

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(self.binary, connectivity=8)
        print(num_labels)
        areas = list()
        for i in range(num_labels):
            areas.append(stats[i][-1])
            # print("轮廓%d的面积:%d" % (i, stats[i][-1]))
        k = 2
        area_avg = np.average(areas[1:-1]) * k
        print("轮廓平均面积:", '%.2f' % area_avg)

        d = z = z1 = 0
        image_filtered = np.zeros_like(self.binary)
        for (i, label) in enumerate(np.unique(labels)):
            # 如果是背景，忽略
            if label == 0:
                continue
            if stats[i][-1] > area_avg:
                image_filtered[labels == i] = 255
                d = d + 1  # 数量
                n = stats[i][2] * stats[i][2] + stats[i][3] * stats[i][3]  # 勾股定理
                x = math.sqrt(n) * 1.1  # 每条长度 1.1为系数
                z = z + x  # 累加总长度
                y = stats[i][-1] / x  # 宽度
                z1 = z1 + y  # 累加总宽度

        print("数量:", d)
        print("平均长度:", '%.2f' % (z / d))
        print("平均宽度:", '%.2f' % (z1 / d))
        image_filtered = 255 - image_filtered
        p2 = 0
        for k in range(0, 270):
            for n in range(0, 270):
                if image_filtered[k, n].all() == 0:
                    p2 += 1
        print("占有率:", '%.2f' % (p2 / 270 / 270))

        rows, columns = image_filtered.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要⽤Format_Indexed8
        QImg = QImage(image_filtered.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_5.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_5.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.label_6.setText("轮廓平均面积:" + str('%.2f' % area_avg) + "\n"
                             + "数量:" + str('%.2f' % d) + "\n"
                             + "平均长度:" + str('%.2f' % (z / d)) + "\n"
                             + "平均宽度:" + str('%.2f' % (z1 / d)) + "\n"
                             + "占有率:" + str('%.2f' % (p2 / 270 / 270))
                             )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())

# # 加载图片
# img = cv.imread('wjl1.jpg')
# img = cv.resize(img, (360, 360), interpolation=cv.INTER_AREA)
# # 灰度化
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # 二值化
# thresh1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 15)
# thresh = 255 - thresh1
#
# # 寻找连通域
# num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
#
# # 计算平均面积
# d = z = z1 = 0
# areas = list()
# for i in range(num_labels):
#     areas.append(stats[i][-1])
#
#     # print("轮廓%d的面积:%d" % (i, stats[i][-1]))
#
# k = 2  # 轮廓面积系数
# area_avg = np.average(areas[1:-1]) * k
# # print("轮廓平均面积:", area_avg)
#
# # 筛选超过平均面积的连通域
# image_filtered = np.zeros_like(img)
# for (i, label) in enumerate(np.unique(labels)):
#     # 如果是背景，忽略
#     if label == 0:
#         continue
#     if stats[i][-1] > area_avg:
#         image_filtered[labels == i] = 255
#         d = d + 1  # 数量
#         n = stats[i][2] * stats[i][2] + stats[i][3] * stats[i][3]  # 勾股定理
#         x = math.sqrt(n) * 1.1  # 每条长度 1.1为系数
#         z = z + x  # 累加总长度
#         y = stats[i][-1] / x  # 宽度
#         z1 = z1 + y  # 累加总宽度
#
# print("数量:", d)
# print("平均长度:", '%.2f' % (z / d))
# print("平均宽度:", '%.2f' % (z1 / d))
#
# skeleton0 = morphology.skeletonize(image_filtered)
# skeleton = skeleton0.astype(np.uint8)
#
# cv.imshow("skeleton.png", skeleton)
#
# # skeleton5 = cv.adaptiveThreshold(skeleton,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,15)
# # contours, hierarchy = cv.findContours(image=skeleton, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
# # length = cv.arcLength(curve=contours[0],closed=True)
# # print('轮廓周长: {}'.format(length))
#
# image_filtered = 255 - image_filtered
#
# p2 = 0
# for k in range(0, 360):
#     for n in range(0, 360):
#         if image_filtered[k, n].all() == 0:
#             p2 += 1
# print("占有率:", '%.2f' % (p2 / 360 / 360))
#
# cv.imshow("image_filtered", image_filtered)
# cv.imshow("img", img_gray)
# cv.imshow("erzhihua", thresh1)
# cv.waitKey()
# cv.destroyAllWindows()
