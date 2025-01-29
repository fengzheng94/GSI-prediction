# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1408, 867)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ico.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.OpenImage = QtWidgets.QPushButton(self.centralwidget)
        self.OpenImage.setGeometry(QtCore.QRect(1210, 190, 131, 51))
        self.OpenImage.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.OpenImage.setObjectName("OpenImage")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(320, 10, 270, 270))
        self.label_2.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_2.setObjectName("label_2")
        self.GrayingImg = QtWidgets.QPushButton(self.centralwidget)
        self.GrayingImg.setGeometry(QtCore.QRect(1210, 250, 131, 51))
        self.GrayingImg.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.GrayingImg.setObjectName("GrayingImg")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(610, 10, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(900, 10, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_4.setObjectName("label_4")
        self.binaryzationImg = QtWidgets.QPushButton(self.centralwidget)
        self.binaryzationImg.setGeometry(QtCore.QRect(1210, 310, 131, 51))
        self.binaryzationImg.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.binaryzationImg.setObjectName("binaryzationImg")
        self.fenxing = QtWidgets.QPushButton(self.centralwidget)
        self.fenxing.setGeometry(QtCore.QRect(1210, 430, 131, 51))
        self.fenxing.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.fenxing.setObjectName("fenxing")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 290, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_5.setObjectName("label_5")
        self.GSI = QtWidgets.QPushButton(self.centralwidget)
        self.GSI.setGeometry(QtCore.QRect(1210, 490, 131, 51))
        self.GSI.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.GSI.setObjectName("GSI")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(610, 290, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_6.setObjectName("label_6")
        self.Road = QtWidgets.QPushButton(self.centralwidget)
        self.Road.setGeometry(QtCore.QRect(1210, 610, 131, 51))
        self.Road.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.Road.setObjectName("Road")
        self.juzhen = QtWidgets.QPushButton(self.centralwidget)
        self.juzhen.setGeometry(QtCore.QRect(1210, 550, 131, 51))
        self.juzhen.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.juzhen.setObjectName("juzhen")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(320, 290, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_7.setObjectName("label_7")
        self.label_231 = QtWidgets.QLabel(self.centralwidget)
        self.label_231.setGeometry(QtCore.QRect(1180, 130, 221, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_231.setFont(font)
        self.label_231.setStyleSheet("color:rgb(0, 0, 255)")
        self.label_231.setObjectName("label_231")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(900, 290, 270, 270))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.label_8.setWordWrap(True)
        self.label_8.setObjectName("label_8")
        self.juzhen_2 = QtWidgets.QPushButton(self.centralwidget)
        self.juzhen_2.setGeometry(QtCore.QRect(1210, 670, 131, 51))
        self.juzhen_2.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.juzhen_2.setObjectName("juzhen_2")
        self.graphicsView = QChartView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 560, 581, 281))
        self.graphicsView.setObjectName("graphicsView")
        self.out = QtWidgets.QPushButton(self.centralwidget)
        self.out.setGeometry(QtCore.QRect(1210, 730, 131, 51))
        self.out.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.out.setObjectName("out")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(1220, 20, 111, 111))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("ico.png"))
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.graphicsView_2 = QChartView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(600, 560, 581, 281))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.quzao = QtWidgets.QPushButton(self.centralwidget)
        self.quzao.setGeometry(QtCore.QRect(1210, 370, 131, 51))
        self.quzao.setStyleSheet("background-color:rgb(0, 170, 0);color:rgb(255, 255, 255)")
        self.quzao.setObjectName("quzao")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能截割"))
        self.OpenImage.setText(_translate("MainWindow", "打开图片"))
        self.label.setText(_translate("MainWindow", "原始图像"))
        self.label_2.setText(_translate("MainWindow", "灰度图"))
        self.GrayingImg.setText(_translate("MainWindow", "灰度化"))
        self.label_3.setText(_translate("MainWindow", "裂隙图"))
        self.label_4.setText(_translate("MainWindow", "分形维数"))
        self.binaryzationImg.setText(_translate("MainWindow", "裂隙提取"))
        self.fenxing.setText(_translate("MainWindow", "分形维数"))
        self.label_5.setText(_translate("MainWindow", "地质强度指标"))
        self.GSI.setText(_translate("MainWindow", "地质强度指标"))
        self.label_6.setText(_translate("MainWindow", "截割路径"))
        self.Road.setText(_translate("MainWindow", "截割路径"))
        self.juzhen.setText(_translate("MainWindow", "GSI矩阵"))
        self.label_7.setText(_translate("MainWindow", "GSI矩阵"))
        self.label_231.setText(_translate("MainWindow", "掘进机智能截割"))
        self.label_8.setText(_translate("MainWindow", "截割数组"))
        self.juzhen_2.setText(_translate("MainWindow", "路径GSI值"))
        self.out.setText(_translate("MainWindow", "输出折线图"))
        self.quzao.setText(_translate("MainWindow", "噪声去除"))
from PyQt5.QtChart import QChartView
