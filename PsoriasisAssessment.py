from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtCore import pyqtSignal
from PIL import ImageOps, ImageEnhance
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.io import imread
import os
import math
from keras.applications.mobilenet import MobileNet
import numpy as np
from keras.utils.np_utils import to_categorical
import scipy.io as sio
import h5py
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.models import Sequential, Model
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import keras
import cv2
from keras.utils.generic_utils import CustomObjectScope
import sys
from sklearn import mixture
from scipy.stats import multivariate_normal
from sklearn.externals import joblib
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from poolinglayers import _GlobalTrimmedAveragePool, TrimmedAveragePool
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from skimage import measure
import os



try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

imgCrop=''
croppedFname=''
global currentPath
currentPath=os.getcwd()
currentPath=currentPath+'/'

class correctionWindow(QtGui.QMainWindow):
    def __init__(self):
        super(correctionWindow, self).__init__()
        self.view2=MyView()
        self.setCentralWidget(self.view2)
        self.setObjectName(_fromUtf8("MainWindow"))
        self.setGeometry(0, 0 , 1366, 710)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 955, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuShow = QtGui.QMenu(self.menubar)
        self.menuShow.setObjectName(_fromUtf8("menuShow"))
        self.menuMark = QtGui.QMenu(self.menubar)
        self.menuMark.setObjectName(_fromUtf8("menuMark"))
        self.menuApply = QtGui.QMenu(self.menubar)
        self.menuApply.setObjectName(_fromUtf8("menuApply"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)
        self.actionEpidermis = QtGui.QAction(self)
        self.actionEpidermis.setObjectName(_fromUtf8("actionEpidermis"))
        self.actionEpidermis.triggered.connect(self.openEpidermis)
        self.actionDermis = QtGui.QAction(self)
        self.actionDermis.setObjectName(_fromUtf8("actionDermis"))
        self.actionDermis.triggered.connect(self.openDermis)
        self.actionNon_Tissue = QtGui.QAction(self)
        self.actionNon_Tissue.setObjectName(_fromUtf8("actionNon_Tissue"))
        self.actionNon_Tissue.triggered.connect(self.openNonTissue)
        self.actionEpidermis_2 = QtGui.QAction(self)
        self.actionEpidermis_2.setObjectName(_fromUtf8("actionEpidermis_2"))
        self.actionEpidermis_2.triggered.connect(self.makeEpidermis)
        self.actionDermis_2 = QtGui.QAction(self)
        self.actionDermis_2.setObjectName(_fromUtf8("actionDermis_2"))
        self.actionDermis_2.triggered.connect(self.makeDermis)
        self.actionNon_Tissue_2 = QtGui.QAction(self)
        self.actionNon_Tissue_2.setObjectName(_fromUtf8("actionNon_Tissue_2"))
        self.actionNon_Tissue_2.triggered.connect(self.makeNonTissue)
        self.actionApply_changes = QtGui.QAction(self)
        self.actionApply_changes.setObjectName(_fromUtf8("actionApply_changes"))
        self.actionApply_changes.triggered.connect(self.applyChanges)
        self.menuShow.addAction(self.actionEpidermis)
        self.menuShow.addAction(self.actionDermis)
        self.menuShow.addAction(self.actionNon_Tissue)
        self.menuMark.addAction(self.actionEpidermis_2)
        self.menuMark.addAction(self.actionDermis_2)
        self.menuMark.addAction(self.actionNon_Tissue_2)
        self.menuApply.addAction(self.actionApply_changes)
        self.menubar.addAction(self.menuShow.menuAction())
        self.menubar.addAction(self.menuMark.menuAction())
        self.menubar.addAction(self.menuApply.menuAction())
        self.imgOpen=0
        self.retranslateUi()
        
    def openEpidermis(self):
        global currentPath
        self.view2.choice=1
        self.imgOpen=1
        self.scene=QtGui.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/epidermis.jpg'))
        self.view2.setScene(self.scene)
        self.scene.setSceneRect(QtCore.QRectF())
        self.view2.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        
    def openDermis(self):
        global currentPath
        self.view2.choice=1
        self.imgOpen=1
        self.scene=QtGui.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/dermis.jpg'))
        self.view2.setScene(self.scene)
        self.scene.setSceneRect(QtCore.QRectF())
        self.view2.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
    def openNonTissue(self):
        global currentPath
        self.view2.choice=1
        self.imgOpen=1
        self.scene=QtGui.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/nontissue.jpg'))
        self.view2.setScene(self.scene)
        self.scene.setSceneRect(QtCore.QRectF())
        self.view2.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def makeEpidermis(self):
        if self.imgOpen==0:
            None
        else:
            global d
            x1=self.view2.x1
            x2=self.view2.x2
            y1=self.view2.y1
            y2=self.view2.y2
            c=d.shape
            aspect=float(c[1])/c[0]
            scale=float(c[0])/660
            width=int(660*aspect)
            width=int(width/2)
            left_column=683-width
            x_1=int((x1-left_column)*scale)
            x_2=int((x2-left_column)*scale)
            y_1=int(y1*scale)
            y_2=int(y2*scale)
            if x_1>x_2:
                t=x_1
                x_1=x_2
                x_2=t
            if y_1>y_2:
                t=y_1
                y_1=y_2
                y_2=t
            if x_1<0:
                x_1=0
            d[y_1:y_2, x_1:x_2]=2
         
    def makeDermis(self):
        if self.imgOpen==0:
            None
        else:
            global d
            x1=self.view2.x1
            x2=self.view2.x2
            y1=self.view2.y1
            y2=self.view2.y2
            c=d.shape
            aspect=float(c[1])/c[0]
            scale=float(c[0])/660
            width=int(660*aspect)
            width=int(width/2)
            left_column=683-width
            x_1=int((x1-left_column)*scale)
            x_2=int((x2-left_column)*scale)
            y_1=int(y1*scale)
            y_2=int(y2*scale)
            if x_1>x_2:
                t=x_1
                x_1=x_2
                x_2=t
            if y_1>y_2:
                t=y_1
                y_1=y_2
                y_2=t
            if x_1<0:
                x_1=0
            d[y_1:y_2, x_1:x_2]=1
            
             
    def makeNonTissue(self):
        if self.imgOpen==0:
            None
        else:
            global d
            x1=self.view2.x1
            x2=self.view2.x2
            y1=self.view2.y1
            y2=self.view2.y2
            c=d.shape
            aspect=float(c[1])/c[0]
            scale=float(c[0])/660
            width=int(660*aspect)
            width=int(width/2)
            left_column=683-width
            x_1=int((x1-left_column)*scale)
            x_2=int((x2-left_column)*scale)
            y_1=int(y1*scale)
            y_2=int(y2*scale)
            if x_1>x_2:
                t=x_1
                x_1=x_2
                x_2=t
            if y_1>y_2:
                t=y_1
                y_1=y_2
                y_2=t
            if x_1<0:
                x_1=0
            d[y_1:y_2, x_1:x_2]=0
             
    def applyChanges(self):
        global SW
        global filename
        global currentPath
        img1=cv2.imread(str(filename))
        img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2=cv2.imread(str(filename))
        img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3=cv2.imread(str(filename))
        img3=cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img11=img1
        img22=img2
        img33=img3
        global d
        d1=d
        d2=d
        d3=d
        [row1,col1]=np.where(d1==2)
        epiMask=np.zeros(d.shape)
        epiMask[row1,col1]=1
        epiMask=cv2.resize(epiMask, ((img11.shape)[1], (img11.shape)[0]))
        epiMask=epiMask*255
        (thresh1, im_bw1) = cv2.threshold(epiMask, 127, 255, 0)
        im_bw1=np.uint8(im_bw1)
        im21, contours1, hierarchy1 = cv2.findContours(im_bw1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img11=cv2.drawContours(img11, contours1, -1, (0,255,0), 7)
        im1=Image.fromarray(img11)
        im1.save(currentPath+'temporary/epidermis.jpg')
        [row2,col2]=np.where(d2==1)
        derMask=np.zeros(d.shape)
        derMask[row2,col2]=1
        derMask=cv2.resize(derMask, ((img22.shape)[1], (img22.shape)[0]))
        derMask=derMask*255
        (thresh2, im_bw2) = cv2.threshold(derMask, 127, 255, 0)
        im_bw2=np.uint8(im_bw2)
        im22, contours2, hierarchy2 = cv2.findContours(im_bw2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img22=cv2.drawContours(img22, contours2, -1, (0,255,0), 7)
        im2=Image.fromarray(img22)
        im2.save(currentPath+'temporary/dermis.jpg')
        [row3,col3]=np.where(d3==0)
        nontMask=np.zeros(d.shape)
        nontMask[row3,col3]=1
        nontMask=cv2.resize(nontMask, ((img33.shape)[1], (img33.shape)[0]))
        nontMask=nontMask*255
        (thresh3, im_bw3) = cv2.threshold(nontMask, 127, 255, 0)
        im_bw3=np.uint8(im_bw3)
        im23, contours3, hierarchy3 = cv2.findContours(im_bw3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img33=cv2.drawContours(img33, contours3, -1, (0,255,0), 7)
        im3=Image.fromarray(img33)
        im3.save(currentPath+'temporary/nontissue.jpg')
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/epidermis.jpg'))
        SW.graphicsView.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/dermis.jpg'))
        SW.graphicsView_2.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView_2.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/nontissue.jpg'))
        SW.graphicsView_3.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView_3.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.close()
    	
    	

    def retranslateUi(self):
        self.setWindowTitle(_translate("MainWindow", "Segmentation Correction", None))
        self.menuShow.setTitle(_translate("MainWindow", "Show", None))
        self.menuMark.setTitle(_translate("MainWindow", "Mark selected as", None))
        self.menuApply.setTitle(_translate("MainWindow", "Apply", None))
        self.actionEpidermis.setText(_translate("MainWindow", "Epidermis", None))
        self.actionDermis.setText(_translate("MainWindow", "Dermis", None))
        self.actionNon_Tissue.setText(_translate("MainWindow", "Non-Tissue", None))
        self.actionEpidermis_2.setText(_translate("MainWindow", "Epidermis", None))
        self.actionDermis_2.setText(_translate("MainWindow", "Dermis", None))
        self.actionNon_Tissue_2.setText(_translate("MainWindow", "Non-Tissue", None))
        self.actionApply_changes.setText(_translate("MainWindow", "Apply changes", None))

class DiseasedWindow(QtGui.QMainWindow):
    def __init__(self):
        super(DiseasedWindow, self).__init__()
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(1366, 720)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.graphicsView = QtGui.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 1362, 660))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 993, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuOptions = QtGui.QMenu(self.menubar)
        self.menuOptions.setObjectName(_fromUtf8("menuOptions"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)
        self.actionDo_corrections = QtGui.QAction(self)
        self.actionDo_corrections.setObjectName(_fromUtf8("actionDo_corrections"))
        self.actionDo_corrections.triggered.connect(self.doCorrections)
        self.actionSave_current_image = QtGui.QAction(self)
        self.actionSave_current_image.setObjectName(_fromUtf8("actionSave_current_image"))
        self.actionSave_current_image.triggered.connect(self.saveCurrent)
        self.menuOptions.addAction(self.actionDo_corrections)
        self.menuOptions.addAction(self.actionSave_current_image)
        self.menubar.addAction(self.menuOptions.menuAction())

        self.retranslateUi()
        
    def doCorrections(self):
        self.CW2=correctionWindow2()
        self.CW2.show()
        
    def saveCurrent(self):    
        global currentPath
        fname1=QtGui.QFileDialog.getSaveFileName(None, 'SaveFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        c=str(fname1)
        img=cv2.imread(currentPath+'temporary/diseasedDetectedFromModel.jpg')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im=Image.fromarray(img)
        im.save(c)
        

    def retranslateUi(self):
        self.setWindowTitle(_translate("MainWindow", "Diseased Region", None))
        self.menuOptions.setTitle(_translate("MainWindow", "Options", None))
        self.actionDo_corrections.setText(_translate("MainWindow", "Do corrections", None))
        self.actionSave_current_image.setText(_translate("MainWindow", "Save current image", None))

class correctionWindow2(QtGui.QMainWindow):
    def __init__(self):
        super(correctionWindow2, self).__init__()
        self.view3=MyView()
        self.setCentralWidget(self.view3)
        self.setObjectName(_fromUtf8("MainWindow"))
        self.setGeometry(0, 0 , 1366, 710)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 955, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuShow = QtGui.QMenu(self.menubar)
        self.menuShow.setObjectName(_fromUtf8("menuShow"))
        self.menuMark = QtGui.QMenu(self.menubar)
        self.menuMark.setObjectName(_fromUtf8("menuMark"))
        self.menuApply = QtGui.QMenu(self.menubar)
        self.menuApply.setObjectName(_fromUtf8("menuApply"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)
        self.actionImage = QtGui.QAction(self)
        self.actionImage.setObjectName(_fromUtf8("actionImage"))
        self.actionImage.triggered.connect(self.openImage)
        self.actionDiseased = QtGui.QAction(self)
        self.actionDiseased.setObjectName(_fromUtf8("actionDiseased"))
        self.actionDiseased.triggered.connect(self.makeDiseased)
        self.actionNonDiseased = QtGui.QAction(self)
        self.actionNonDiseased.setObjectName(_fromUtf8("actionNonDiseased"))
        self.actionNonDiseased.triggered.connect(self.makeNonDiseased)
        self.actionApply_changes = QtGui.QAction(self)
        self.actionApply_changes.setObjectName(_fromUtf8("actionApply_changes"))
        self.actionApply_changes.triggered.connect(self.applyChanges)
        self.menuShow.addAction(self.actionImage)
        self.menuMark.addAction(self.actionDiseased)
        self.menuMark.addAction(self.actionNonDiseased)
        self.menuApply.addAction(self.actionApply_changes)
        self.menubar.addAction(self.menuShow.menuAction())
        self.menubar.addAction(self.menuMark.menuAction())
        self.menubar.addAction(self.menuApply.menuAction())
        self.retranslateUi()
        self.imgOpen=0
        
    def openImage(self):
        global currentPath
        self.view3.choice=1
        self.imgOpen=1
        self.scene=QtGui.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/diseasedDetectedFromModel.jpg'))
        self.view3.setScene(self.scene)
        self.scene.setSceneRect(QtCore.QRectF())
        self.view3.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
       

    def makeDiseased(self):
        if self.imgOpen==0:
            None
        else:
            global DisMask
            x1=self.view3.x1
            x2=self.view3.x2
            y1=self.view3.y1
            y2=self.view3.y2
            c=DisMask.shape
            aspect=float(c[1])/c[0]
            scale=float(c[0])/660
            width=int(660*aspect)
            width=int(width/2)
            left_column=683-width
            x_1=int((x1-left_column)*scale)
            x_2=int((x2-left_column)*scale)
            y_1=int(y1*scale)
            y_2=int(y2*scale)
            if x_1>x_2:
                t=x_1
                x_1=x_2
                x_2=t
            if y_1>y_2:
                t=y_1
                y_1=y_2
                y_2=t
            if x_1<0:
                x_1=0
            DisMask[y_1:y_2, x_1:x_2]=255
         
    def makeNonDiseased(self):
        if self.imgOpen==0:
            None
        else:
            global DisMask
            x1=self.view3.x1
            x2=self.view3.x2
            y1=self.view3.y1
            y2=self.view3.y2
            c=DisMask.shape
            aspect=float(c[1])/c[0]
            scale=float(c[0])/660
            width=int(660*aspect)
            width=int(width/2)
            left_column=683-width
            x_1=int((x1-left_column)*scale)
            x_2=int((x2-left_column)*scale)
            y_1=int(y1*scale)
            y_2=int(y2*scale)
            if x_1>x_2:
                t=x_1
                x_1=x_2
                x_2=t
            if y_1>y_2:
                t=y_1
                y_1=y_2
                y_2=t
            if x_1<0:
                x_1=0
            un, counts=np.unique(DisMask, return_counts=True)
            print(un)
            print(counts)
            DisMask[y_1:y_2, x_1:x_2]=0
            
             
    def applyChanges(self):
        global DW
        global imgCrop
        global DisMask
        global currentPath
        img=cv2.imread(str(imgCrop))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=img
        DisMask=cv2.resize(DisMask, ((img2.shape)[1], (img2.shape)[0]))
        _, DisMask=cv2.threshold(DisMask, 127, 255, 0)
        DisMask=np.uint8(DisMask)	
        un, counts=np.unique(DisMask, return_counts=True)
        print(un)
        print(counts)
        im21, contours1, hierarchy1 = cv2.findContours(DisMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img=cv2.drawContours(img, contours1, -1, (0,255,0), 2)
        im=Image.fromarray(img)
        im.save(currentPath+'temporary/diseasedDetectedFromModel.jpg')
        DW.scene=QtGui.QGraphicsScene()
        DW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/diseasedDetectedFromModel.jpg'))
        DW.graphicsView.setScene(DW.scene)
        DW.scene.setSceneRect(QtCore.QRectF())
        DW.graphicsView.fitInView(DW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.close()
    	
    	

    def retranslateUi(self):
        self.setWindowTitle(_translate("MainWindow", "Segmentation Correction", None))
        self.menuShow.setTitle(_translate("MainWindow", "Show", None))
        self.menuMark.setTitle(_translate("MainWindow", "Mark selected as", None))
        self.menuApply.setTitle(_translate("MainWindow", "Apply", None))
        self.actionImage.setText(_translate("MainWindow", "Open Image", None))
        self.actionDiseased.setText(_translate("MainWindow", "Diseased", None))
        self.actionNonDiseased.setText(_translate("MainWindow", "Non-Diseased", None))
        self.actionApply_changes.setText(_translate("MainWindow", "Apply changes", None))
        
        
class UniformityReport(QtGui.QMainWindow):
    def __init__(self, max_len, min_len, notrap):
    	super(UniformityReport, self).__init__()
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(400, 200)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel("The maximum thickness of the Epidermis is "+str(max_len)+" pixels", self)
        self.label.setGeometry(QtCore.QRect(50, 30, 320, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel("The minimum thickness of the Epidermis is "+str(min_len)+" pixels", self)
        self.label_2.setGeometry(QtCore.QRect(50, 80, 320, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel("The total number of trapped dermis regions are "+str(notrap), self)
        self.label_3.setGeometry(QtCore.QRect(50, 130, 300, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()

    def retranslateUi(self):
        self.setWindowTitle(_translate("MainWindow", "Histopathology Report", None))
        
      
class UniformitySCReport(QtGui.QMainWindow):
    def __init__(self, max_len, min_len):
    	super(UniformitySCReport, self).__init__()
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(460, 228)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel("The maximum thickness of the Stratum Corneum is "+str(max_len)+" pixels", self)
        self.label.setGeometry(QtCore.QRect(50, 50, 360, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel("The minimum thickness of the Stratum Corneum is "+str(min_len)+" pixels", self)
        self.label_2.setGeometry(QtCore.QRect(50, 130, 360, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()

    def retranslateUi(self):
        self.setWindowTitle(_translate("MainWindow", "Stratum Corneum Uniformity Report", None))

class munroMicroabscessDialog(QtGui.QDialog):
    def __init__(self, regions):
    	super(munroMicroabscessDialog, self).__init__()
        self.setObjectName(_fromUtf8("Dialog"))
        self.resize(490, 528)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(80, 450, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label = QtGui.QLabel("There is a total of "+str(regions)+" regions detected to have Munro's microabscess", self)
        self.label.setGeometry(QtCore.QRect(50, 30, 390, 31))
        self.label.setObjectName(_fromUtf8("label"))
        self.checkBox = QtGui.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(90, 110, 241, 22))
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.checkBox.stateChanged.connect(self.checkBox1State)
        self.checkBox_2 = QtGui.QCheckBox(self)
        self.checkBox_2.setGeometry(QtCore.QRect(90, 160, 231, 22))
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.checkBox_2.stateChanged.connect(self.checkBox2State)
        self.checkBox_3 = QtGui.QCheckBox(self)
        self.checkBox_3.setGeometry(QtCore.QRect(90, 210, 221, 22))
        self.checkBox_3.setObjectName(_fromUtf8("checkBox_3"))
        self.checkBox_3.stateChanged.connect(self.checkBox3State)
        self.checkBox_4 = QtGui.QCheckBox(self)
        self.checkBox_4.setGeometry(QtCore.QRect(90, 260, 191, 22))
        self.checkBox_4.setObjectName(_fromUtf8("checkBox_4"))
        self.checkBox_4.stateChanged.connect(self.checkBox4State)
        self.checkBox_5 = QtGui.QCheckBox(self)
        self.checkBox_5.setGeometry(QtCore.QRect(90, 310, 241, 22))
        self.checkBox_5.stateChanged.connect(self.checkBox5State)
        self.checkBox_5.setObjectName(_fromUtf8("checkBox_5"))
        self.checkBox_6 = QtGui.QCheckBox(self)
        self.checkBox_6.setGeometry(QtCore.QRect(90, 360, 241, 22))
        self.checkBox_6.stateChanged.connect(self.checkBox6State)
        self.checkBox_6.setObjectName(_fromUtf8("checkBox_6"))
        self.checkBox_7 = QtGui.QCheckBox(self)
        self.checkBox_7.setGeometry(QtCore.QRect(90, 410, 360, 22))
        self.checkBox_7.setObjectName(_fromUtf8("checkBox_7"))
        self.returnVal=0
	
        self.retranslateUi()
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), self.ok)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), self.cancel)
        
    def ok(self):
    	self.returnVal=1
    	self.close()
    	
    def cancel(self):
    	self.returnVal=0
    	self.close()
    	
    def checkBox1State(self):
    	if self.checkBox.isChecked():
    		self.checkBox_2.setChecked(False)
    		self.checkBox_3.setChecked(False)
    		self.checkBox_4.setChecked(False)
    		self.checkBox_5.setChecked(False)
    		self.checkBox_6.setChecked(False)
    
    
    def checkBox2State(self):
    	if self.checkBox_2.isChecked():
    		self.checkBox.setChecked(False)
    		self.checkBox_3.setChecked(False)
    		self.checkBox_4.setChecked(False)
    		self.checkBox_5.setChecked(False)
    		self.checkBox_6.setChecked(False)
    
    
    def checkBox3State(self):
    	if self.checkBox_3.isChecked():
    		self.checkBox.setChecked(False)
    		self.checkBox_2.setChecked(False)
    		self.checkBox_4.setChecked(False)
    		self.checkBox_5.setChecked(False)
    		self.checkBox_6.setChecked(False)
    
    
    def checkBox4State(self):
        if self.checkBox_4.isChecked():
    		self.checkBox.setChecked(False)
    		self.checkBox_2.setChecked(False)
    		self.checkBox_3.setChecked(False)
    		self.checkBox_5.setChecked(False)
    		self.checkBox_6.setChecked(False)
    
    
    def checkBox5State(self):
    	if self.checkBox_5.isChecked():
    		self.checkBox.setChecked(False)
    		self.checkBox_2.setChecked(False)
    		self.checkBox_3.setChecked(False)
    		self.checkBox_4.setChecked(False)
    		self.checkBox_6.setChecked(False)
    		
    def checkBox6State(self):
    	if self.checkBox_6.isChecked():
    		self.checkBox.setChecked(False)
    		self.checkBox_2.setChecked(False)
    		self.checkBox_3.setChecked(False)
    		self.checkBox_4.setChecked(False)
    		self.checkBox_5.setChecked(False)
    		
  
    def retranslateUi(self):
        self.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.checkBox.setText(_translate("Dialog", "Show top 5 regions", None))
        self.checkBox_2.setText(_translate("Dialog", "Show top 10 regions", None))
        self.checkBox_3.setText(_translate("Dialog", "Show top 15 regions", None))
        self.checkBox_4.setText(_translate("Dialog", "Show top 20 regions", None))
        self.checkBox_5.setText(_translate("Dialog", "Show top 25 regions", None))
        self.checkBox_6.setText(_translate("Dialog", "Show all", None))
        self.checkBox_7.setText(_translate("Dialog", "Get thickness details of Stratum Corneum", None))


class SegmentationWindow(QtGui.QTabWidget):
    	
    def __init__(self):
        super(SegmentationWindow, self).__init__()
        self.setObjectName(_fromUtf8("TabWidget"))
        self.setGeometry(0, 0 , 1366, 710)
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.graphicsView = QtGui.QGraphicsView(self.tab)
        self.graphicsView.setGeometry(QtCore.QRect(5, 1, 1353, 628))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.pushButton = QtGui.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(1090, 634, 261, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.clicked.connect(self.uniformReport)
        self.pushButton_4 = QtGui.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(850, 634, 200, 27))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_4.clicked.connect(self.saveEpidermis)
        self.pushButton_7 = QtGui.QPushButton(self.tab)
        self.pushButton_7.setGeometry(QtCore.QRect(725, 634, 100, 27))
        self.pushButton_7.setObjectName(_fromUtf8("pushButton_7"))
        self.pushButton_7.clicked.connect(self.doCorrection)
        self.addTab(self.tab, _fromUtf8(""))
        self.tab1 = QtGui.QWidget()
        self.tab1.setObjectName(_fromUtf8("tab1"))
        self.graphicsView_2 = QtGui.QGraphicsView(self.tab1)
        self.graphicsView_2.setGeometry(QtCore.QRect(5, 1, 1353, 628))
        self.graphicsView_2.setObjectName(_fromUtf8("graphicsView_2"))
        self.pushButton_2 = QtGui.QPushButton(self.tab1)
        self.pushButton_2.setGeometry(QtCore.QRect(1090, 634, 261, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.clicked.connect(self.uniformReport)
        self.pushButton_5 = QtGui.QPushButton(self.tab1)
        self.pushButton_5.setGeometry(QtCore.QRect(850, 634, 200, 27))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.pushButton_5.clicked.connect(self.saveDermis)
        self.pushButton_8 = QtGui.QPushButton(self.tab1)
        self.pushButton_8.setGeometry(QtCore.QRect(725, 634, 100, 27))
        self.pushButton_8.setObjectName(_fromUtf8("pushButton_8"))
        self.pushButton_8.clicked.connect(self.doCorrection)
        self.addTab(self.tab1, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.graphicsView_3 = QtGui.QGraphicsView(self.tab_2)
        self.graphicsView_3.setGeometry(QtCore.QRect(5, 1, 1353, 628))
        self.graphicsView_3.setObjectName(_fromUtf8("graphicsView_3"))
        self.pushButton_3 = QtGui.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(1090, 634, 261, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_3.clicked.connect(self.uniformReport)
        self.pushButton_6 = QtGui.QPushButton(self.tab_2)
        self.pushButton_6.setGeometry(QtCore.QRect(850, 634, 200, 27))
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_5"))
        self.pushButton_6.clicked.connect(self.saveNonTissue)
        self.pushButton_9 = QtGui.QPushButton(self.tab_2)
        self.pushButton_9.setGeometry(QtCore.QRect(725, 634, 100, 27))
        self.pushButton_9.setObjectName(_fromUtf8("pushButton_9"))
        self.pushButton_9.clicked.connect(self.doCorrection)
        
        self.addTab(self.tab_2, _fromUtf8(""))

        self.retranslateUi()
        self.setCurrentIndex(0)
        
    def uniformReport(self):
        global d
        global filename
        img11=cv2.imread(str(filename))
        [row,col]=np.where(d==2)
        epiMask=np.zeros(d.shape)
        epiMask[row,col]=1
        epiMask=cv2.resize(epiMask, ((img11.shape)[1], (img11.shape)[0]))
        epiMask=epiMask*255
        (thresh1, im_bw1) = cv2.threshold(epiMask, 127, 255, 0)
        im_bw1=np.uint8(im_bw1)
    	lengthMax=0
        lengthMin=(im_bw1.shape)[1]
        for i in range((im_bw1.shape)[0]):
        	row=im_bw1[i,:]
        	index=np.argwhere(row==255)
        	c=index.shape
        	if c[0]!=0:
        		left=int(index[0])
        		right=int(index[(c[0]-1)])
        		for a in range(c[0]-1):
        			if (index[a+1]-index[a])>300:
        				left=index[a+1]
        		j=c[0]-1
        		while j>=0:
        			if index[j]-index[j-1]>300:
        				right=index[j-1]
        			j=j-1
        		length=right-left
        	else:
        		continue
        	if length>50:
        		if lengthMax<length:
        			lengthMax=length
        		if lengthMin>length:
        			lengthMin=length
        lengthMax=int(lengthMax)
        lengthMin=int(lengthMin)
        numberTrapped=0
        kernel = np.ones((15, 15),np.uint8)
        [row, col]=np.where(d==2)
        [row1, col1]=np.where(d==0)
        Mask=np.ones(d.shape)
        Mask[row, col]=0
        Mask[row1, col1]=0
        un, counts=np.unique(Mask, return_counts=True)
        Mask=np.uint8(Mask)
        Mask=cv2.erode(Mask, kernel, iterations = 1)
        blobs_labels = measure.label(Mask, background=0)
        c, counts=np.unique(blobs_labels, return_counts=True)
        c1=c.shape
        for i in range(c1[0]):
        	if counts[i]>200:
        		numberTrapped=numberTrapped+1
        	elif counts[i]<200:
        		continue
        numberTrapped=numberTrapped-2
    	self.UR=UniformityReport(lengthMax, lengthMin, numberTrapped)
    	self.UR.show()
    	
    def saveEpidermis(self):
        global currentPath
    	fname1=QtGui.QFileDialog.getSaveFileName(None, 'SaveFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        c=str(fname1)
        img=cv2.imread(currentPath+'temporary/epidermis.jpg')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im=Image.fromarray(img)
        im.save(c)
        
    def saveDermis(self):
        global currentPath
    	fname1=QtGui.QFileDialog.getSaveFileName(None, 'SaveFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        c=str(fname1)
        img=cv2.imread(currentPath+'temporary/dermis.jpg')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im=Image.fromarray(img)
        im.save(c)
        
    def saveNonTissue(self):
        global currentPath
    	fname1=QtGui.QFileDialog.getSaveFileName(None, 'SaveFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        c=str(fname1)
        img=cv2.imread(currentPath+'temporary/nontissue.jpg')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im=Image.fromarray(img)
        im.save(c)
        
    def doCorrection(self):
    	self.CW=correctionWindow()
    	self.CW.imgOpen=0
    	self.CW.show()
        
        
    def retranslateUi(self):
        self.setWindowTitle(_translate("TabWidget", "Segmentation", None))
        self.setTabText(self.indexOf(self.tab), _translate("TabWidget", "Epidermis", None))
        self.setTabText(self.indexOf(self.tab1), _translate("TabWidget", "Dermis", None))
        self.setTabText(self.indexOf(self.tab_2), _translate("TabWidget", "Non-Tissue", None))
        self.pushButton.setText(_translate("TabWidget", "Get Histopathology Report", None))
        self.pushButton_2.setText(_translate("TabWidget", "Get Histopathology Report", None))
        self.pushButton_3.setText(_translate("TabWidget", "Get Histopathology Report", None))
        self.pushButton_4.setText(_translate("TabWidget", "Save Epidermis Image", None))
        self.pushButton_5.setText(_translate("TabWidget", "Save Dermis Image", None))
        self.pushButton_6.setText(_translate("TabWidget", "Save Non-Tissue Image", None))
        self.pushButton_7.setText(_translate("TabWidget", "Do corrections", None))
        self.pushButton_8.setText(_translate("TabWidget", "Do corrections", None))
        self.pushButton_9.setText(_translate("TabWidget", "Do corrections", None))


class Classification(QtGui.QDialog):

    def __init__(self):
        super(Classification, self).__init__()
        self.setObjectName(_fromUtf8("Dialog"))
        self.resize(695, 590)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(340, 540, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.graphicsView = QtGui.QGraphicsView(self)
        self.graphicsView.setGeometry(QtCore.QRect(5, 11, 671, 371))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.scene=QtGui.QGraphicsScene()
        self.checkBox = QtGui.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(60, 390, 121, 41))
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.checkBox_2 = QtGui.QCheckBox(self)
        self.checkBox_2.setGeometry(QtCore.QRect(60, 440, 111, 41))
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.checkBox_3 = QtGui.QCheckBox(self)
        self.checkBox_3.setGeometry(QtCore.QRect(60, 490, 131, 41))
        self.checkBox_3.setObjectName(_fromUtf8("checkBox_3"))
        self.lineEdit = QtGui.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(350, 400, 211, 27))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.lineEdit_2 = QtGui.QLineEdit(self)
        self.lineEdit_2.setGeometry(QtCore.QRect(350, 450, 211, 27))
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.lineEdit_3 = QtGui.QLineEdit(self)
        self.lineEdit_3.setGeometry(QtCore.QRect(350, 500, 211, 27))
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.pushButton = QtGui.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(590, 400, 85, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.clicked.connect(self.erythemaModel)
        self.pushButton_2 = QtGui.QPushButton(self)
        self.pushButton_2.setGeometry(QtCore.QRect(590, 450, 85, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.clicked.connect(self.scalingModel)
        self.pushButton_3 = QtGui.QPushButton(self)
        self.pushButton_3.setGeometry(QtCore.QRect(590, 500, 85, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_3.clicked.connect(self.indurationModel)
        self.label = QtGui.QLabel(self)
        self.label.setGeometry(QtCore.QRect(245, 396, 91, 31))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(242, 450, 101, 31))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(245, 506, 91, 21))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.returnVal=0

        self.retranslateUi()
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), self.ok)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), self.cancel)
    
    def ok(self):
        self.returnVal=1
        self.close()

    def cancel(self):
        self.returnVal=0
        self.close()
    
    def erythemaModel(self):
        fname=QtGui.QFileDialog.getOpenFileName(None, 'Browse','',"Model(*.h5)")
        self.lineEdit.setText(fname)
    
    def scalingModel(self):
        fname=QtGui.QFileDialog.getOpenFileName(None, 'Browse','',"Model(*.h5)")
        self.lineEdit_2.setText(fname) 

    def indurationModel(self):
        fname=QtGui.QFileDialog.getOpenFileName(None, 'Browse','',"Model(*.h5)")
        self.lineEdit_3.setText(fname)        
    
    def retranslateUi(self):
        self.setWindowTitle(_translate("Dialog", "Severity Assesment", None))
        self.checkBox.setText(_translate("Dialog", "Erythema Score", None))
        self.checkBox_2.setText(_translate("Dialog", "Scaling Score", None))
        self.checkBox_3.setText(_translate("Dialog", "Induration Score", None))
        self.pushButton.setText(_translate("Dialog", "Browse", None))
        self.pushButton_2.setText(_translate("Dialog", "Browse", None))
        self.pushButton_3.setText(_translate("Dialog", "Browse", None))
        self.label.setText(_translate("Dialog", "Choose Model :", None))
        self.label_2.setText(_translate("Dialog", " Choose Model :", None))
        self.label_3.setText(_translate("Dialog", "Choose Model :", None))


class MyView(QtGui.QGraphicsView):
    rectChanged = pyqtSignal(QtCore.QRect)

    def __init__(self):
        super(MyView, self).__init__()
        #QtGui.QGraphicsView.__init__(self)
        self.setGeometry(20, 30, 100, 200)
        self.scene = QtGui.QGraphicsScene(self)
        self.resize(1326, 400) 
        self.setScene(self.scene)
        self._pixmapHandle = None
        self.x1=0
        self.y1=0
        self.x2=0
        self.y2=0
        self.choice=2
        self.rubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)
        self.setMouseTracking(True)
        self.origin = QtCore.QPoint()
        self.changeRubberBand = False
        self.rubberBand.move(0, 0)


        # Image aspect ratio mode.
        # !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Stack of QRectF zoom boxes in scene coordinates.
        self.zoomStack = []

        # Flags for enabling/disabling mouse interaction.
        self.canZoom = True
        self.canPan = True

    def save(Self, filename, imgarray):
        if (len(imgarray.shape))==3:
            im=Image.fromarray(imgarray, 'RGB')
        else:
            im=Image.fromarray(imgarray)
        im.save(filename)

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.rubberBand.resize(self.size())

    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        self.x1=event.x()
        self.y1=event.y()
        global zoom
        if event.button() == Qt.LeftButton:
            if self.canPan:
            	self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        elif event.button() == Qt.RightButton and self.choice==0:
            if zoom>0:
                for i in range(zoom):
                    self.scale(0.91, 0.91)
                zoom=0
            self.origin = event.pos()
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rectChanged.emit(self.rubberBand.geometry())
            self.rubberBand.show()
            self.changeRubberBand = True
        elif event.button() == Qt.RightButton and self.choice==1:
            if zoom>0:
                for i in range(zoom):
                    self.scale(0.91, 0.91)
                zoom=0
            self.origin = event.pos()
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rectChanged.emit(self.rubberBand.geometry())
            self.rubberBand.show()
            self.changeRubberBand = True
        QtGui.QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.changeRubberBand:
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())
            self.rectChanged.emit(self.rubberBand.geometry())
        QtGui.QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """   
        global currentPath 
        self.x2=event.x()
        self.y2=event.y()  
        QtGui.QGraphicsView.mouseReleaseEvent(self, event)
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.setDragMode(QtGui.QGraphicsView.NoDrag)
        elif event.button() == Qt.RightButton and self.choice==0:
            global croppedFname
            global filename
            img=cv2.imread(str(filename))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            c=img.shape
            aspect=float(c[1])/c[0]
            scale=float(c[0])/660
            width=int(660*aspect)
            width=int(width/2)
            left_column=683-width
            self.changeRubberBand = False
            currentQRect = self.rubberBand.geometry()
            x_1=int((self.x1-left_column)*scale)
            x_2=int((self.x2-left_column)*scale)
            y_1=int(self.y1*scale)
            y_2=int(self.y2*scale)
            if x_1>x_2:
                t=x_1
                x_1=x_2
                x_2=t
            if y_1>y_2:
                t=y_1
                y_1=y_2
                y_2=t
            if x_1<0:
                x_1=0 
            img_cropped=img[y_1:y_2, x_1:x_2, :]
            if self.x1==self.x2 or self.y1==self.y2:
                croppedFname=''
            else:
                imz=Image.fromarray(img_cropped)
                imz.save(currentPath+'temporary/cropped.jpg')
                croppedFname=currentPath+'temporary/cropped.jpg'
            self.setDragMode(QtGui.QGraphicsView.NoDrag)
        elif event.button() == Qt.RightButton and self.choice==1:
            self.changeRubberBand = False
            currentQRect = self.rubberBand.geometry()
            self.setDragMode(QtGui.QGraphicsView.NoDrag)

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        if event.button() == Qt.LeftButton:
            None
        elif event.button() == Qt.RightButton:
            None
        QtGui.QGraphicsView.mouseDoubleClickEvent(self, event)

        
class ThirdWindow(QtGui.QMainWindow): 

    def __init__(self, h, w, bpp): 
        super(ThirdWindow, self).__init__() 
        self.resize(400, 120)
        self.setWindowTitle("Image Information")
        lbl=QtGui.QLabel('Height : '+h+' pixels', self)
        lbl.setGeometry(10, 20, 350, 30)
        lbl1=QtGui.QLabel('Width : '+w+' pixels', self)
        lbl1.setGeometry(10, 50, 350, 30)
        lbl2=QtGui.QLabel('Color Depth : '+bpp+' channels', self) 
        lbl2.setGeometry(10, 80, 350, 30)

class ReportWindow(QtGui.QMainWindow): 

    def __init__(self, e, s, i): 
        super(ReportWindow, self).__init__() 
        self.setWindowTitle("Psoriasis Report")
        if e==0 and s==0:
            self.resize(450, 100)
            self.lbl2=QtGui.QLabel('On a scale of 0 to 3, the person has an Induration level of '+str(i), self)
            self.lbl2.setGeometry(10, 40, 400, 30)
        elif e==0 and i==0:
            self.resize(450, 100)
            self.lbl2=QtGui.QLabel('On a scale of 0 to 4, the person has an Scaling level of '+str(s), self)
            self.lbl2.setGeometry(10, 40, 400, 30)
        elif i==0 and s==0:
            self.resize(450, 100)
            self.lbl2=QtGui.QLabel('On a scale of 0 to 4, the person has an Erythema level of '+str(e), self)
            self.lbl2.setGeometry(10, 40, 400, 30)
        elif e==0:
            self.resize(450, 100)
            self.lbl=QtGui.QLabel('On a scale of 0 to 4, the person has an Scaling level of '+str(s), self)
            self.lbl.setGeometry(10, 20, 400, 30)
            self.lbl1=QtGui.QLabel('On a scale of 0 to 3, the person has an Induration level of '+str(i), self)
            self.lbl1.setGeometry(10, 70, 400, 30)
        elif s==0:
            self.resize(450, 100)
            self.lbl=QtGui.QLabel('On a scale of 0 to 4, the person has an Erythema level of '+str(e), self)
            self.lbl.setGeometry(10, 20, 400, 30)
            self.lbl1=QtGui.QLabel('On a scale of 0 to 3, the person has an Induration level of '+str(i), self)
            self.lbl1.setGeometry(10, 70, 400, 30)
        elif i==0:
            self.resize(450, 100)
            self.lbl=QtGui.QLabel('On a scale of 0 to 4, the person has an Erythema level of '+str(e), self)
            self.lbl.setGeometry(10, 20, 400, 30)
            self.lbl1=QtGui.QLabel('On a scale of 0 to 4, the person has an Scaling level of '+str(s), self)
            self.lbl1.setGeometry(10, 70, 400, 30)
        else:
            self.resize(450, 150)
            self.lbl=QtGui.QLabel('On a scale of 0 to 4, the person has an Erythema level of '+str(e), self)
            self.lbl.setGeometry(10, 20, 400, 30)
            self.lbl1=QtGui.QLabel('On a scale of 0 to 4, the person has an Scaling level of '+str(s), self)
            self.lbl1.setGeometry(10, 70, 400, 30)
            self.lbl2=QtGui.QLabel('On a scale of 0 to 3, the person has an Induration level of '+str(i), self)
            self.lbl2.setGeometry(10, 120, 400, 30)
        
        
class AdvancedOptions(QtGui.QDialog):

    def __init__(self):
        super(AdvancedOptions, self).__init__()
        self.setObjectName(_fromUtf8("Dialog"))
        self.resize(514, 385)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(110, 340, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.line = QtGui.QFrame(self)
        self.line.setGeometry(QtCore.QRect(220, 10, 20, 321))
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.groupBox = QtGui.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 191, 331))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.checkBox = QtGui.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(20, 40, 131, 22))
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.checkBox.stateChanged.connect(self.stateChange1)
        self.checkBox_2 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_2.setGeometry(QtCore.QRect(20, 80, 121, 22))
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.checkBox_2.stateChanged.connect(self.stateChange2)
        self.checkBox_3 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_3.setGeometry(QtCore.QRect(20, 120, 111, 22))
        self.checkBox_3.setObjectName(_fromUtf8("checkBox_3"))
        self.checkBox_3.stateChanged.connect(self.stateChange3)
        self.checkBox_4 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_4.setGeometry(QtCore.QRect(20, 160, 91, 22))
        self.checkBox_4.setObjectName(_fromUtf8("checkBox_4"))
        self.checkBox_4.stateChanged.connect(self.stateChange4)
        self.checkBox_5 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_5.setGeometry(QtCore.QRect(20, 200, 151, 22))
        self.checkBox_5.setObjectName(_fromUtf8("checkBox_5"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 240, 56, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.checkBox_6 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_6.setGeometry(QtCore.QRect(20, 260, 88, 22))
        self.checkBox_6.setObjectName(_fromUtf8("checkBox_6"))
        self.checkBox_6.stateChanged.connect(self.stateChange6)
        self.checkBox_7 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_7.setGeometry(QtCore.QRect(80, 260, 88, 22))
        self.checkBox_7.setObjectName(_fromUtf8("checkBox_7"))
        self.checkBox_7.stateChanged.connect(self.stateChange7)
        self.checkBox_8 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_8.setGeometry(QtCore.QRect(140, 260, 88, 22))
        self.checkBox_8.setObjectName(_fromUtf8("checkBox_8"))
        self.checkBox_8.stateChanged.connect(self.stateChange8)
        self.checkBox_9 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_9.setGeometry(QtCore.QRect(50, 290, 88, 22))
        self.checkBox_9.setObjectName(_fromUtf8("checkBox_9"))
        self.checkBox_9.stateChanged.connect(self.stateChange9)
        self.checkBox_10 = QtGui.QCheckBox(self.groupBox)
        self.checkBox_10.setGeometry(QtCore.QRect(110, 290, 88, 22))
        self.checkBox_10.setObjectName(_fromUtf8("checkBox_10"))
        self.checkBox_10.stateChanged.connect(self.stateChange10)
        self.groupBox_2 = QtGui.QGroupBox(self)
        self.groupBox_2.setGeometry(QtCore.QRect(240, 20, 261, 311))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.checkBox_11 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_11.setGeometry(QtCore.QRect(10, 40, 161, 22))
        self.checkBox_11.setObjectName(_fromUtf8("checkBox_11"))
        self.checkBox_12 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_12.setGeometry(QtCore.QRect(10, 80, 101, 22))
        self.checkBox_12.setObjectName(_fromUtf8("checkBox_12"))
        self.lineEdit = QtGui.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(110, 80, 51, 27))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.label_2 = QtGui.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(160, 80, 56, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.checkBox_13 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_13.setGeometry(QtCore.QRect(10, 130, 88, 22))
        self.checkBox_13.setObjectName(_fromUtf8("checkBox_13"))
        self.lineEdit_2 = QtGui.QLineEdit(self.groupBox_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(90, 130, 51, 27))
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.label_3 = QtGui.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(150, 130, 71, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.checkBox_14 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_14.setGeometry(QtCore.QRect(10, 190, 141, 22))
        self.checkBox_14.setObjectName(_fromUtf8("checkBox_14"))
        self.lineEdit_3 = QtGui.QLineEdit(self.groupBox_2)
        self.lineEdit_3.setGeometry(QtCore.QRect(150, 190, 51, 27))
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.label_4 = QtGui.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(200, 190, 61, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.checkBox_15 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_15.setGeometry(QtCore.QRect(10, 240, 88, 22))
        self.checkBox_15.setObjectName(_fromUtf8("checkBox_15"))
        self.lineEdit_4 = QtGui.QLineEdit(self.groupBox_2)
        self.lineEdit_4.setGeometry(QtCore.QRect(100, 240, 51, 27))
        self.lineEdit_4.setObjectName(_fromUtf8("lineEdit_4"))
        self.label_5 = QtGui.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(160, 240, 56, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.checkBox_16 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_16.setGeometry(QtCore.QRect(10, 280, 111, 22))
        self.checkBox_16.setObjectName(_fromUtf8("checkBox_16"))
        self.lineEdit_5 = QtGui.QLineEdit(self.groupBox_2)
        self.lineEdit_5.setGeometry(QtCore.QRect(120, 280, 51, 27))
        self.lineEdit_5.setObjectName(_fromUtf8("lineEdit_5"))
        self.label_6 = QtGui.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(180, 280, 56, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))

        self.retranslateUi()
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), self.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), self.reject)

    def stateChange1(self):
        if self.checkBox.isChecked():
            self.checkBox_2.setChecked(False)
    
    def stateChange2(self):
        if self.checkBox_2.isChecked():
            self.checkBox.setChecked(False)

    def stateChange3(self):
        if self.checkBox_3.isChecked():
            self.checkBox_4.setChecked(False)

    def stateChange4(self):
        if self.checkBox_4.isChecked():
            self.checkBox_3.setChecked(False)   
    
    def stateChange6(self):
        if self.checkBox_6.isChecked():
            self.checkBox_5.setChecked(False)
            self.checkBox_7.setChecked(False)
            self.checkBox_8.setChecked(False) 
            self.checkBox_9.setChecked(False)
            self.checkBox_10.setChecked(False)

    def stateChange7(self):
        if self.checkBox_7.isChecked():
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            self.checkBox_8.setChecked(False) 
            self.checkBox_9.setChecked(False)
            self.checkBox_10.setChecked(False)

    def stateChange8(self):
        if self.checkBox_8.isChecked():
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            self.checkBox_7.setChecked(False) 
            self.checkBox_9.setChecked(False)
            self.checkBox_10.setChecked(False)

    def stateChange9(self):
        if self.checkBox_9.isChecked():
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            self.checkBox_7.setChecked(False) 
            self.checkBox_8.setChecked(False)
            self.checkBox_10.setChecked(False)

    def stateChange10(self):
        if self.checkBox_10.isChecked():
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            self.checkBox_7.setChecked(False) 
            self.checkBox_8.setChecked(False)
            self.checkBox_9.setChecked(False)


    def retranslateUi(self):
        self.setWindowTitle(_translate("Dialog", "Advanced Options", None))
        self.groupBox.setTitle(_translate("Dialog", "Transformations", None))
        self.checkBox.setText(_translate("Dialog", "Horizontal flip", None))
        self.checkBox_2.setText(_translate("Dialog", "Vertical flip", None))
        self.checkBox_3.setText(_translate("Dialog", "Right rotate", None))
        self.checkBox_4.setText(_translate("Dialog", "Left rotate", None))
        self.checkBox_5.setText(_translate("Dialog", "Convert to grayscale", None))
        self.label.setText(_translate("Dialog", "RGB to :", None))
        self.checkBox_6.setText(_translate("Dialog", "RBG", None))
        self.checkBox_7.setText(_translate("Dialog", "BGR", None))
        self.checkBox_8.setText(_translate("Dialog", "BRG", None))
        self.checkBox_9.setText(_translate("Dialog", "GRB", None))
        self.checkBox_10.setText(_translate("Dialog", "GBR", None))
        self.groupBox_2.setTitle(_translate("Dialog", "Miscellaneous", None))
        self.checkBox_11.setText(_translate("Dialog", "Preserve aspect ratio", None))
        self.checkBox_12.setText(_translate("Dialog", "Brightness :", None))
        self.label_2.setText(_translate("Dialog", "(0-5)", None))
        self.checkBox_13.setText(_translate("Dialog", "Contrast :", None))
        self.label_3.setText(_translate("Dialog", "(0-25)", None))
        self.checkBox_14.setText(_translate("Dialog", "Gamma correction :", None))
        self.label_4.setText(_translate("Dialog", "(0.01-6.99)", None))
        self.checkBox_15.setText(_translate("Dialog", "Blur filter :", None))
        self.label_5.setText(_translate("Dialog", "(1-99)", None))
        self.checkBox_16.setText(_translate("Dialog", "Median filter :", None))
        self.label_6.setText(_translate("Dialog", "(3-9)", None))

class ThresholdDialog(QtGui.QDialog):

    def __init__(self):
        super(ThresholdDialog, self).__init__()
        self.setObjectName(_fromUtf8("Dialog"))
        self.resize(400, 194)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(30, 140, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.lineEdit = QtGui.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(180, 40, 113, 27))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.label = QtGui.QLabel(self)
        self.label.setGeometry(QtCore.QRect(35, 45, 141, 20))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(310, 45, 56, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.returnVal=0

        self.retranslateUi()
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), self.ok)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), self.cancel)

    def ok(self):
        self.returnVal=1
        self.close()
    
    def cancel(self):
        self.returnVal=0
        self.close()

    def retranslateUi(self):
        self.setWindowTitle(_translate("Dialog", "Enter Threshold Value", None))
        self.label.setText(_translate("Dialog", "Enter Threshold value :", None))
        self.label_2.setText(_translate("Dialog", "( 0 - 255 )", None))

class AdapThreshold(QtGui.QDialog):
    def __init__(self):
        super(AdapThreshold, self).__init__()
        self.setObjectName(_fromUtf8("Dialog"))
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(30, 230, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.checkBox = QtGui.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(80, 40, 261, 22))
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.checkBox.setChecked(True)
        self.checkBox.stateChanged.connect(self.checkBox1)
        self.checkBox_2 = QtGui.QCheckBox(self)
        self.checkBox_2.setGeometry(QtCore.QRect(80, 90, 251, 22))
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.checkBox_2.setChecked(False)
        self.checkBox_2.stateChanged.connect(self.checkBox2)
        self.label = QtGui.QLabel(self)
        self.label.setGeometry(QtCore.QRect(55, 150, 71, 20))
        self.label.setObjectName(_fromUtf8("label"))
        self.lineEdit = QtGui.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(160, 140, 113, 27))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.label_2 = QtGui.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(300, 150, 56, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.returnVal=0

        self.retranslateUi()
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), self.ok)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), self.cancel)

    def ok(self):
        self.returnVal=1
        self.close()
    
    def cancel(self):
        self.returnVal=0
        self.close()

    def checkBox1(self):
        if self.checkBox.isChecked():
            self.checkBox_2.setChecked(False)
    
    def checkBox2(self):
        if self.checkBox_2.isChecked():
            self.checkBox.setChecked(False)
                

    def retranslateUi(self):
        self.setWindowTitle(_translate("Dialog", "Adaptive Threshold", None))
        self.checkBox.setText(_translate("Dialog", "Mean of the Values in the Block Size", None))
        self.checkBox_2.setText(_translate("Dialog", "Weighted Mean of Gaussian Window", None))
        self.label.setText(_translate("Dialog", "Block Size :", None))
        self.label_2.setText(_translate("Dialog", "( 5 - 20 )", None))


class Ui_Dialog(QtGui.QDialog):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        self.setObjectName(_fromUtf8("Dialog"))
        self.resize(715, 573)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(350, 530, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.groupBox = QtGui.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(29, 20, 231, 111))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.checkBox = QtGui.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(10, 20, 211, 22))
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.checkBox.setChecked(True)
        self.checkBox.stateChanged.connect(self.changeStateResize1)
        self.lineEdit = QtGui.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(80, 40, 113, 21))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.lineEdit.setDisabled(False)
        self.lineEdit_2 = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_2.setGeometry(QtCore.QRect(80, 70, 113, 21))
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.lineEdit_2.setDisabled(False)
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 40, 56, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 70, 56, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.groupBox_2 = QtGui.QGroupBox(self)
        self.groupBox_2.setGeometry(QtCore.QRect(370, 20, 241, 111))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.checkBox_2 = QtGui.QCheckBox(self.groupBox_2)
        self.checkBox_2.setGeometry(QtCore.QRect(20, 20, 211, 22))
        self.checkBox_2.setObjectName(_fromUtf8("checkBox_2"))
        self.checkBox_2.stateChanged.connect(self.changeStateResize2)
        self.lineEdit_3 = QtGui.QLineEdit(self.groupBox_2)
        self.lineEdit_3.setGeometry(QtCore.QRect(110, 56, 113, 31))
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.lineEdit_3.setDisabled(True)
        self.label_3 = QtGui.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(30, 60, 71, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.groupBox_3 = QtGui.QGroupBox(self)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 140, 611, 121))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.checkBox_3 = QtGui.QCheckBox(self.groupBox_3)
        self.checkBox_3.setGeometry(QtCore.QRect(20, 20, 211, 22))
        self.checkBox_3.setObjectName(_fromUtf8("checkBox_3"))
        self.checkBox_3.setChecked(True)
        self.checkBox_3.stateChanged.connect(self.changeName)
        self.lineEdit_4 = QtGui.QLineEdit(self.groupBox_3)
        self.lineEdit_4.setGeometry(QtCore.QRect(140, 50, 113, 31))
        self.lineEdit_4.setObjectName(_fromUtf8("lineEdit_4"))
        self.label_4 = QtGui.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(40, 60, 91, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(45, 90, 301, 20))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.lineEdit_5 = QtGui.QLineEdit(self.groupBox_3)
        self.lineEdit_5.setGeometry(QtCore.QRect(480, 6, 113, 31))
        self.lineEdit_5.setObjectName(_fromUtf8("lineEdit_5"))
        self.lineEdit_6 = QtGui.QLineEdit(self.groupBox_3)
        self.lineEdit_6.setGeometry(QtCore.QRect(480, 66, 113, 31))
        self.lineEdit_6.setObjectName(_fromUtf8("lineEdit_6"))
        self.label_6 = QtGui.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(355, 10, 111, 20))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_7 = QtGui.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(360, 70, 71, 17))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.line = QtGui.QFrame(self)
        self.line.setGeometry(QtCore.QRect(30, 120, 641, 20))
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.line_2 = QtGui.QFrame(self)
        self.line_2.setGeometry(QtCore.QRect(330, 20, 20, 111))
        self.line_2.setFrameShape(QtGui.QFrame.VLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.line_3 = QtGui.QFrame(self)
        self.line_3.setGeometry(QtCore.QRect(30, 260, 631, 20))
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.groupBox_4 = QtGui.QGroupBox(self)
        self.groupBox_4.setGeometry(QtCore.QRect(60, 290, 261, 101))
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.checkBox_4 = QtGui.QCheckBox(self.groupBox_4)
        self.checkBox_4.setGeometry(QtCore.QRect(20, 20, 141, 22))
        self.checkBox_4.setObjectName(_fromUtf8("checkBox_4"))
        self.checkBox_4.setChecked(True)
        self.checkBox_4.stateChanged.connect(self.changeFormat)
        self.comboBox = QtGui.QComboBox(self.groupBox_4)
        self.comboBox.setGeometry(QtCore.QRect(150, 40, 101, 33))
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.label_8 = QtGui.QLabel(self.groupBox_4)
        self.label_8.setGeometry(QtCore.QRect(50, 50, 91, 17))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.line_4 = QtGui.QFrame(self)
        self.line_4.setGeometry(QtCore.QRect(340, 270, 20, 141))
        self.line_4.setFrameShape(QtGui.QFrame.VLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.line_5 = QtGui.QFrame(self)
        self.line_5.setGeometry(QtCore.QRect(40, 400, 631, 20))
        self.line_5.setFrameShape(QtGui.QFrame.HLine)
        self.line_5.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_5.setObjectName(_fromUtf8("line_5"))
        self.groupBox_5 = QtGui.QGroupBox(self)
        self.groupBox_5.setGeometry(QtCore.QRect(370, 290, 261, 101))
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.lineEdit_7 = QtGui.QLineEdit(self.groupBox_5)
        self.lineEdit_7.setGeometry(QtCore.QRect(10, 30, 251, 27))
        self.lineEdit_7.setObjectName(_fromUtf8("lineEdit_7"))
        self.pushButton = QtGui.QPushButton(self.groupBox_5)
        self.pushButton.setGeometry(QtCore.QRect(150, 70, 85, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.clicked.connect(self.browse)    
        self.pushButton_2 = QtGui.QPushButton(self)
        self.pushButton_2.setGeometry(QtCore.QRect(414, 450, 201, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.clicked.connect(self.advancedOptions)
        self.line_6 = QtGui.QFrame(self)
        self.line_6.setGeometry(QtCore.QRect(340, 410, 20, 121))
        self.line_6.setFrameShape(QtGui.QFrame.VLine)
        self.line_6.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_6.setObjectName(_fromUtf8("line_6"))
        self.groupBox_6 = QtGui.QGroupBox(self)
        self.groupBox_6.setGeometry(QtCore.QRect(60, 420, 261, 125))
        self.groupBox_6.setObjectName(_fromUtf8("groupBox_6"))
        self.textEdit_8 = QtGui.QTextEdit(self.groupBox_6)
        self.textEdit_8.setGeometry(QtCore.QRect(10, 30, 251, 50))
        self.textEdit_8.setObjectName(_fromUtf8("lineEdit_8"))
        self.pushButton_3 = QtGui.QPushButton(self.groupBox_6)
        self.pushButton_3.setGeometry(QtCore.QRect(140, 90, 120, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_3.clicked.connect(self.browseOpen)
        self.pushButton_4 = QtGui.QPushButton(self.groupBox_6)
        self.pushButton_4.setGeometry(QtCore.QRect(60, 90, 70, 27))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_4.clicked.connect(self.addFiles)
        self.pushButton_5 = QtGui.QPushButton(self.groupBox_6)
        self.pushButton_5.setGeometry(QtCore.QRect(5, 90, 45, 27))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_5.clicked.connect(self.removeAll)
        self.fnameSave=''
        self.fnameOpen=''
        self.fileNames=[]
        self.returnVal=0
        self.ao=AdvancedOptions()

        self.retranslateUi()
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), self.ok)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), self.cancel)

    def ok(self):
        self.returnVal=1
        self.close()

    def cancel(self):
        self.returnVal=0
        self.close()

    def advancedOptions(self):
        self.ao.exec_()
    
    def changeStateResize1(self):
        if self.checkBox.isChecked():
            self.checkBox_2.setChecked(False)
            self.lineEdit.setDisabled(False)
            self.lineEdit_2.setDisabled(False)
        else:
            self.lineEdit.setDisabled(True)
            self.lineEdit_2.setDisabled(True)
            

    def changeStateResize2(self):
        if  self.checkBox_2.isChecked():
            self.checkBox.setChecked(False)
            self.lineEdit_3.setDisabled(False)
        else:
            self.lineEdit_3.setDisabled(True)
                
    def changeName(self):
        if self.checkBox_3.isChecked():
            self.lineEdit_4.setDisabled(False)
            self.lineEdit_5.setDisabled(False)
            self.lineEdit_6.setDisabled(False)
        else:
            self.lineEdit_4.setDisabled(True)
            self.lineEdit_5.setDisabled(True)
            self.lineEdit_6.setDisabled(True)

    def changeFormat(self):
        if self.checkBox_4.isChecked():
            self.comboBox.setDisabled(False)
        else:
            self.comboBox.setDisabled(True)

    def browse(self):
        fname=QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        if fname!='':
            self.fnameSave=fname
            self.lineEdit_7.setText(self.fnameSave)

    def browseOpen(self):
        fname=QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        if fname!='':
            self.fnameOpen=fname
            self.textEdit_8.setText(self.fnameOpen)
    
    def addFiles(self):
        caption = 'Open files'
        # use current/working directory
        directory = './'
        filter_mask = "Image files (*.jpg *.jpeg *.png *.bmp *.gif *.tif)"
        options = QtGui.QFileDialog.Options()
        fnames = [str(f) for f in QtGui.QFileDialog.getOpenFileNames(None, caption, directory, filter_mask, options=options)]
        self.fileNames=fnames
        for fn in fnames:
            self.textEdit_8.append(fn)

    def removeAll(self):
        self.fileNames=[]
        self.fnameOpen=''
        self.textEdit_8.clear()
        

    def retranslateUi(self):
        self.setWindowTitle(_translate("Dialog", "Batch Processing", None))
        self.groupBox.setTitle(_translate("Dialog", "Resize using number of pixels", None))
        self.lineEdit.setText(_translate("Dialog", "100", None))
        self.lineEdit_2.setText(_translate("Dialog", "100", None))
        self.checkBox.setText(_translate("Dialog", "Resize using pixel specifications", None))
        self.label.setText(_translate("Dialog", "Width :", None))
        self.label_2.setText(_translate("Dialog", "Height :", None))
        self.groupBox_2.setTitle(_translate("Dialog", "Resize as percentage of original", None))
        self.checkBox_2.setText(_translate("Dialog", "Resize as percentage of original ", None))
        self.lineEdit_3.setText(_translate("Dialog", "25", None))
        self.label_3.setText(_translate("Dialog", "Percentage:", None))
        self.groupBox_3.setTitle(_translate("Dialog", "Rename", None))
        self.checkBox_3.setText(_translate("Dialog", "Rename while Batch Processing", None))
        self.lineEdit_4.setText(_translate("Dialog", "image#####", None))
        self.lineEdit_5.setText(_translate("Dialog", "1", None))
        self.lineEdit_6.setText(_translate("Dialog", "1", None))
        self.label_4.setText(_translate("Dialog", "Name Format :", None))
        self.label_5.setText(_translate("Dialog", "Change number of # to number images accordingly", None))
        self.label_6.setText(_translate("Dialog", "Starting Number :", None))
        self.label_7.setText(_translate("Dialog", "Increment :", None))
        self.groupBox_4.setTitle(_translate("Dialog", "File format conversion", None))
        self.checkBox_4.setText(_translate("Dialog", "Convert file format", None))
        self.comboBox.setItemText(0, _translate("Dialog", "*.jpg (JPG)", None))
        self.comboBox.setItemText(1, _translate("Dialog", "*.jpeg (JPEG)", None))
        self.comboBox.setItemText(2, _translate("Dialog", "*.png (PNG)", None))
        self.comboBox.setItemText(3, _translate("Dialog", "*.bmp (BMP)", None))
        self.comboBox.setItemText(4, _translate("Dialog", "*.tif (TIFF)", None))
        self.comboBox.setItemText(5, _translate("Dialog", "*.gif (GIF)", None))
        self.label_8.setText(_translate("Dialog", "Output format :", None))
        self.groupBox_5.setTitle(_translate("Dialog", "Output directory for result files ", None))
        self.lineEdit_7.setText(_translate("Dialog", "/home/resultant", None))
        self.pushButton.setText(_translate("Dialog", "Browse", None))
        self.pushButton_2.setText(_translate("Dialog", "Advanced Options", None))
        self.groupBox_6.setTitle(_translate("Dialog", "Choose Input Files/Folder", None))
        self.pushButton_3.setText(_translate("Dialog", "Browse for Folder", None))
        self.pushButton_4.setText(_translate("Dialog", "Add Files", None))   
        self.pushButton_5.setText(_translate("Dialog", "Clear", None))

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.view=MyView()
        self.setWindowTitle("Psoriasis Assessment")
        self.setGeometry(0, 0 , 1366, 750)
        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 30))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setObjectName(_fromUtf8("menuEdit"))
        self.menuApplication = QtGui.QMenu(self.menubar)
        self.menuApplication.setObjectName(_fromUtf8("menuApplication"))
        self.menuFiltering = QtGui.QMenu(self.menuEdit)
        self.menuFiltering.setObjectName(_fromUtf8("menuFiltering"))
        self.menuTransform = QtGui.QMenu(self.menuEdit)
        self.menuTransform.setObjectName(_fromUtf8("menuTransform"))
        self.menuFlip = QtGui.QMenu(self.menuTransform)
        self.menuFlip.setObjectName(_fromUtf8("menuFlip"))
        self.menuRotate = QtGui.QMenu(self.menuTransform)
        self.menuRotate.setObjectName(_fromUtf8("menuRotate"))
        self.menuThresholding=QtGui.QMenu(self.menuFile)
        self.menuThresholding.setObjectName(_fromUtf8("menuThresholding"))
        self.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(self)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(self)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionOpen.triggered.connect(self.showDialog)
        self.actionHistogram = QtGui.QAction(self)
        self.actionHistogram.setObjectName(_fromUtf8("actionHistogram"))
        self.actionHistogram.triggered.connect(self.showHistogram)
        self.actionImage_Information = QtGui.QAction(self)
        self.actionImage_Information.setObjectName(_fromUtf8("actionImage_Information"))
        self.actionImage_Information.triggered.connect(self.showImageInformation)
        self.actionBatch_Processing = QtGui.QAction(self)
        self.actionBatch_Processing.setObjectName(_fromUtf8("actionBatch_Processing"))
        self.actionBatch_Processing.triggered.connect(self.BatchProcessing)
        self.actionSlide_Show = QtGui.QAction(self)
        self.actionSlide_Show.setObjectName(_fromUtf8("actionSlide_Show"))
        self.actionSave_As = QtGui.QAction(self)
        self.actionSave_As.setObjectName(_fromUtf8("actionSave_As"))
        self.actionSave_As.triggered.connect(self.saveas)
        self.actionExit = QtGui.QAction(self)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.actionExit.triggered.connect(QtGui.QApplication.quit)
        self.actionUndo = QtGui.QAction(self)
        self.actionUndo.setObjectName(_fromUtf8("actionUndo"))
        self.actionUndo.triggered.connect(self.undo)
        self.actionRedo = QtGui.QAction(self)
        self.actionRedo.setObjectName(_fromUtf8("actionRedo"))
        self.actionRedo.triggered.connect(self.redo)
        self.actionCrop = QtGui.QAction(self)
        self.actionCrop.setObjectName(_fromUtf8("actionCrop"))
        self.actionCrop.triggered.connect(self.crop)
        self.actionColor_channel_decomposition = QtGui.QAction(self)
        self.actionColor_channel_decomposition.setObjectName(_fromUtf8("actionColor_channel_decomposition"))
        self.actionColor_channel_decomposition.triggered.connect(self.colorChannelDecomposition)
        self.actionMean_Filtering = QtGui.QAction(self)
        self.actionMean_Filtering.setObjectName(_fromUtf8("actionMean_Filtering"))
        self.actionMean_Filtering.triggered.connect(self.MeanFiltering)
        self.actionMaximum_Filtering = QtGui.QAction(self)
        self.actionMaximum_Filtering.setObjectName(_fromUtf8("actionMaximum_Filtering"))
        self.actionMaximum_Filtering.triggered.connect(self.MaximumFiltering)
        self.actionMedian_Filtering = QtGui.QAction(self)
        self.actionMedian_Filtering.setObjectName(_fromUtf8("actionMedian_Filtering"))
        self.actionMedian_Filtering.triggered.connect(self.MedianFiltering)
        self.actionContrast = QtGui.QAction(self)
        self.actionContrast.setObjectName(_fromUtf8("actionContrast"))
        self.actionContrast.triggered.connect(self.contrast)
        self.actionThresholding = QtGui.QAction(self)
        self.actionThresholding.setObjectName(_fromUtf8("actionThresholding"))
        self.actionRight = QtGui.QAction(self)
        self.actionRight.setObjectName(_fromUtf8("actionHorizontal"))
        self.actionRight.triggered.connect(self.FlipHorizontal)
        self.actionLeft = QtGui.QAction(self)
        self.actionLeft.setObjectName(_fromUtf8("actionVertical"))
        self.actionLeft.triggered.connect(self.FlipVertical)
        self.actionRight_2 = QtGui.QAction(self)
        self.actionRight_2.setObjectName(_fromUtf8("actionRight_2"))
        self.actionRight_2.triggered.connect(self.RotateRight)
        self.actionLeft_2 = QtGui.QAction(self)
        self.actionLeft_2.setObjectName(_fromUtf8("actionLeft_2"))
        self.actionLeft_2.triggered.connect(self.RotateLeft)
        self.actionSimpleThresholding=QtGui.QAction(self)
        self.actionSimpleThresholding.setObjectName(_fromUtf8("actionSimpleThresholding"))
        self.actionSimpleThresholding.triggered.connect(self.simpleThresholding)
        self.actionOtsu=QtGui.QAction(self)
        self.actionOtsu.setObjectName(_fromUtf8("actionOtsu"))
        self.actionOtsu.triggered.connect(self.otsu)
        self.actionAdaptiveThresholding=QtGui.QAction(self)
        self.actionAdaptiveThresholding.setObjectName(_fromUtf8("actionAdaptiveThresholding"))
        self.actionAdaptiveThresholding.triggered.connect(self.adaptiveThresholding)
        self.actionSeverityAssessment=QtGui.QAction(self)
        self.actionSeverityAssessment.setObjectName(_fromUtf8("actionSeverityAssessment"))
        self.actionSeverityAssessment.triggered.connect(self.severityAssessment)
        self.actionDiseaseDetection=QtGui.QAction(self)
        self.actionDiseaseDetection.setObjectName(_fromUtf8("actionDiseaseDetection"))
        self.actionDiseaseDetection.triggered.connect(self.diseaseDetection)
        self.actionHistopathologySegmentation=QtGui.QAction(self)
        self.actionHistopathologySegmentation.setObjectName(_fromUtf8("actionHistopathologySegmentation"))
        self.actionHistopathologySegmentation.triggered.connect(self.histopathologySegmentation)
        self.actionMunrosMicroabcess=QtGui.QAction(self)
        self.actionMunrosMicroabcess.setObjectName(_fromUtf8("actionMunrosMicroabcess"))
        self.actionMunrosMicroabcess.triggered.connect(self.mm)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionHistogram)
        self.menuFile.addAction(self.actionImage_Information)
        self.menuFile.addAction(self.actionBatch_Processing)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionExit)
        self.menuFiltering.addAction(self.actionMean_Filtering)
        self.menuFiltering.addAction(self.actionMaximum_Filtering)
        self.menuFiltering.addAction(self.actionMedian_Filtering)
        self.menuFlip.addAction(self.actionRight)
        self.menuFlip.addAction(self.actionLeft)
        self.menuRotate.addAction(self.actionRight_2)
        self.menuRotate.addAction(self.actionLeft_2)
        self.menuTransform.addAction(self.menuFlip.menuAction())
        self.menuTransform.addAction(self.menuRotate.menuAction())
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addAction(self.actionCrop)
        self.menuEdit.addAction(self.actionColor_channel_decomposition)
        self.menuEdit.addAction(self.menuFiltering.menuAction())
        self.menuEdit.addAction(self.actionContrast)
        self.menuEdit.addAction(self.menuThresholding.menuAction())
        self.menuEdit.addAction(self.menuTransform.menuAction())
        self.menuApplication.addAction(self.actionSeverityAssessment)
        self.menuApplication.addAction(self.actionDiseaseDetection)
        self.menuApplication.addAction(self.actionHistopathologySegmentation)
        self.menuApplication.addAction(self.actionMunrosMicroabcess)
        self.menuThresholding.addAction(self.actionSimpleThresholding)
        self.menuThresholding.addAction(self.actionOtsu)
        self.menuThresholding.addAction(self.actionAdaptiveThresholding)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuApplication.menuAction())
        self.imgFeed=0
        self.imgBatch=0
        self.img=0
        global filename
        self.fname=''
        global zoom
        zoom=0
        self.retranslateUi()
        self.setCentralWidget(self.view)
        self.undoStack=[]
        self.undoStackSize=0
        self.undoStackSizeCopy=0
        self.redoStackSizeCopy=0
        self.redoStackSize=0
        self.redoStack=[]
        self.width=''
        self.height=''
        self.bp = Ui_Dialog()
        self.cl=Classification()
        global SW
        SW=SegmentationWindow()
        global DW
        DW=DiseasedWindow()
        self.home()

    def home(self):
        global currentPath
        saveAction = QtGui.QAction(QtGui.QIcon(currentPath+'index.jpeg'), 'Save', self)
        saveAction.triggered.connect(self.save)
        self.toolBar = self.addToolBar("Image Operations")
        self.toolBar.addAction(saveAction)
        saveZoom_In = QtGui.QAction(QtGui.QIcon(currentPath+'zoom-in.png'), 'Zoom In', self)
        saveZoom_In.triggered.connect(self.Zoom_In1)
        self.toolBar.addAction(saveZoom_In)
        saveZoom_Out = QtGui.QAction(QtGui.QIcon(currentPath+'zoom-out.png'), 'Zoom Out', self)
        saveZoom_Out.triggered.connect(self.Zoom_Out1)
        self.toolBar.addAction(saveZoom_Out)
        savePrevious_Image = QtGui.QAction(QtGui.QIcon(currentPath+'leftarrow.jpg'), 'Previous Image', self)
        savePrevious_Image.triggered.connect(self.previousImage)
        self.toolBar.addAction(savePrevious_Image)
        saveNext_Image = QtGui.QAction(QtGui.QIcon(currentPath+'rightarrow.jpg'), 'Next Image', self)
        saveNext_Image.triggered.connect(self.nextImage)
        self.toolBar.addAction(saveNext_Image)
        undo = QtGui.QAction(QtGui.QIcon(currentPath+'undo.png'), 'Undo', self)
        undo.triggered.connect(self.undo)
        self.toolBar.addAction(undo)
        redo = QtGui.QAction(QtGui.QIcon(currentPath+'redo.png'), 'Redo', self)
        redo.triggered.connect(self.redo)
        self.toolBar.addAction(redo)
        self.Label=QtGui.QLabel()
        self.Label.move(10, 700)
        self.show()

    def showDialog(self):
        self.view.choice=0
        global filename
        self.scene=QtGui.QGraphicsScene()
        fname=QtGui.QFileDialog.getOpenFileName(None, 'OpenFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        if fname!='':
            self.fname=fname
            filename=fname
        c=str(self.fname)
        global imgCrop
        imgCrop=c
        img1=cv2.imread(c)
        self.img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize    
        self.scene.addPixmap(QtGui.QPixmap(self.fname))
        self.view.setScene(self.scene)
        self.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.cl.scene=QtGui.QGraphicsScene()
        self.cl.scene.addPixmap(QtGui.QPixmap(imgCrop))
        self.cl.graphicsView.setScene(self.cl.scene)
        self.cl.scene.setSceneRect(QtCore.QRectF())
        self.cl.graphicsView.fitInView(self.cl.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        global SW
        global DW
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(self.fname))
        SW.graphicsView.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(self.fname))
        SW.graphicsView_2.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView_2.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(self.fname))
        SW.graphicsView_3.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView_3.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        DW.scene=QtGui.QGraphicsScene()
        DW.scene.addPixmap(QtGui.QPixmap(self.fname))
        DW.graphicsView.setScene(DW.scene)
        DW.scene.setSceneRect(QtCore.QRectF())
        DW.graphicsView.fitInView(DW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        


    def saveas(self):
        fname1=QtGui.QFileDialog.getSaveFileName(None, 'SaveFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        c=str(fname1)
        self.view.save(c, self.img)
     

    def save(self):
        c=str(self.fname)
        self.view.save(c, self.img)
        

    def Zoom_In1(self):
        global zoom
        zoom+=1
        factor=1.10
        self.view.scale(factor, factor)

    def Zoom_Out1(self):
        global zoom
        if zoom>0:
            zoom-=1
            factor=0.91
            self.view.scale(factor, factor)

    def showHistogram(self):
        if (len(self.img.shape))==3:        
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([self.img],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        else:
            histr = cv2.calcHist([self.img],[0], None, [256], [0,256])
            plt.plot(histr)
            plt.xlim([0,256])
            plt.show()
        
    def showImageInformation(self):
        h=''
        w=''
        bpp=''
        if self.fname=='':
            h=''
            w=''
            bpp=''
        else:
            c = np.shape(self.img)
            h=c[0]
            w=c[1]
            bpp=c[2]
            h=str(h)
            w=str(w)
            bpp=str(bpp)
        self.TW = ThirdWindow(h, w, bpp) 
        self.TW.show()
        
    
    def MedianFiltering(self):
        global currentPath
        c=str(self.fname)
        img1=cv2.imread(c)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        self.img=cv2.medianBlur(img2, 3)
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        im=Image.fromarray(self.img, 'RGB')
        im.save(currentPath+'temporary/median.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/median.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def MeanFiltering(self):
        global currentPath
        c=str(self.fname)
        img1=cv2.imread(c)
        img2= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        self.img=cv2.blur(img2, (3,3))
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        im=Image.fromarray(self.img, 'RGB')
        im.save(currentPath+'temporary/mean.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/mean.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def MaximumFiltering(self):
        global currentPath
        c=str(self.fname)
        kernel = np.ones((3,3),np.uint8)
        img1=cv2.imread(c)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        self.img=cv2.dilate(img2, kernel, iterations = 1)
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        im=Image.fromarray(self.img, 'RGB')
        im.save(currentPath+'temporary/max.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/max.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def FlipHorizontal(self):
        global currentPath
        self.img = cv2.flip(self.img, 0)
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        if (len(self.img.shape))==3:
            im=Image.fromarray(self.img, 'RGB')
        else:
            im=Image.fromarray(self.img)
        im.save(currentPath+'temporary/horizontal.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/horizontal.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def FlipVertical(self):
        global currentPath
        self.img = cv2.flip(self.img, 1)
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        if (len(self.img.shape))==3:
            im=Image.fromarray(self.img, 'RGB')
        else:
            im=Image.fromarray(self.img)
        im.save(currentPath+'temporary/vertical.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/vertical.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def RotateRight(self):
        global currentPath
        num_rows = (self.img.shape)[0]
        num_cols = (self.img.shape)[1]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -90, 1)
        self.img = cv2.warpAffine(self.img, rotation_matrix, (num_cols, num_rows))
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        if (len(self.img.shape))==3:
            im=Image.fromarray(self.img, 'RGB')
        else:
            im=Image.fromarray(self.img)
        im.save(currentPath+'temporary/rotateright.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/rotateright.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def RotateLeft(self):
        global currentPath
        num_rows = (self.img.shape)[0]
        num_cols = (self.img.shape)[1]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
        self.img = cv2.warpAffine(self.img, rotation_matrix, (num_cols, num_rows))
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        if (len(self.img.shape))==3:
            im=Image.fromarray(self.img, 'RGB')
        else:
            im=Image.fromarray(self.img)
        im.save(currentPath+'temporary/rotateleft.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/rotateleft.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def colorChannelDecomposition(self):
        c=str(self.fname)
        src = cv2.imread(c)
        bgr=cv2.split(src)  

        cv2.imshow("Blue", bgr[0])
        cv2.imshow("Green", bgr[1])
        cv2.imshow("Red", bgr[2])

    def previousImage(self):
        c=str(self.fname)
        global filename
        d=c.rfind('/')
        path=c[0:d+1]
        imgLst=self.imageFilePaths(path)
        current=imgLst.index(c)
        if current==0:
            self.fname=imgLst[(len(imgLst)-1)]
            filename=imgLst[(len(imgLst)-1)]
        else:
            self.fname=imgLst[current-1]
            filename=imgLst[current-1]
        self.img=cv2.imread(str(self.fname))
        self.img=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(self.fname))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def nextImage(self):
        c=str(self.fname)
        global filename
        d=c.rfind('/')
        path=c[0:d+1]
        imgLst=self.imageFilePaths(path)
        current=imgLst.index(c)
        if current==(len(imgLst)-1):
            self.fname=imgLst[0]
            filename=imgLst[0]
        else:
            self.fname=imgLst[current+1]
            filename=imgLst[current+1]
        self.img=cv2.imread(str(self.fname))
        self.img=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(self.fname))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    
    def imageFilePaths(self, paths):
        text_files = [f for f in os.listdir(paths) if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp') or f.endswith('.jpeg') or f.endswith('.tif') or f.endswith('.gif') or f.endswith('.PNG') or f.endswith('.JPG') or f.endswith('.BMP') or f.endswith('.JPEG') or f.endswith('.TIF') or f.endswith('.GIF'))]
        for i in range(len(text_files)):
            text_files[i]=paths+text_files[i]
        return text_files

    def undo(self):
        global currentPath
        if self.undoStackSize==0:
            None
        elif self.undoStackSizeCopy==len(self.undoStack):
            imgz=self.undoStack.pop()
            self.undoStackSize-=1
            self.redoStack.append(imgz) 
            self.redoStackSize+=1
            img1=self.undoStack.pop()
            self.undoStackSize-=1
            self.redoStack.append(img1)
            self.redoStackSize+=1
            self.redoStackSizeCopy=self.redoStackSize
            self.img=img1
            if (len(self.img.shape))==3:
                im=Image.fromarray(img1, 'RGB')
            else:
                im=Image.fromarray(img1)
            im.save(currentPath+'temporary/undo.png')
            self.view.scene=QtGui.QGraphicsScene(self)
            self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/undo.png'))
            self.view.setScene(self.view.scene)
            self.view.scene.setSceneRect(QtCore.QRectF())
            self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            img1=self.undoStack.pop()
            self.undoStackSize-=1
            self.redoStack.append(img1)
            self.redoStackSize+=1
            self.redoStackSizeCopy=self.redoStackSize
            self.img=img1
            if (len(self.img.shape))==3:
                im=Image.fromarray(img1, 'RGB')
            else:
                im=Image.fromarray(img1)
            im.save(currentPath+'temporary/undo.png')
            self.view.scene=QtGui.QGraphicsScene(self)
            self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/undo.png'))
            self.view.setScene(self.view.scene)
            self.view.scene.setSceneRect(QtCore.QRectF())
            self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def redo(self):
        global currentPath
        if self.redoStackSize==0:
            None
        elif self.redoStackSizeCopy==len(self.redoStack):
            imgz=self.redoStack.pop()
            self.redoStackSize-=1
            self.undoStack.append(imgz) 
            self.undoStackSize+=1
            img1=self.redoStack.pop()
            self.redoStackSize-=1
            self.undoStack.append(img1)
            self.undoStackSize+=1   
            self.undoStackSizeCopy=self.undoStackSize
            self.img=img1
            if (len(self.img.shape))==3:
                im=Image.fromarray(img1, 'RGB')
            else:
                im=Image.fromarray(img1)
            im.save(currentPath+'temporary/redo.png')
            self.view.scene=QtGui.QGraphicsScene(self)
            self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/redo.png'))
            self.view.setScene(self.view.scene)
            self.view.scene.setSceneRect(QtCore.QRectF())
            self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            img1=self.redoStack.pop()
            self.redoStackSize-=1
            self.undoStack.append(img1)
            self.undoStackSize+=1   
            self.undoStackSizeCopy=self.undoStackSize
            self.img=img1
            if (len(self.img.shape))==3:
                im=Image.fromarray(img1, 'RGB')
            else:
                im=Image.fromarray(img1)
            im.save(currentPath+'temporary/redo.png')
            self.view.scene=QtGui.QGraphicsScene(self)
            self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/redo.png'))
            self.view.setScene(self.view.scene)
            self.view.scene.setSceneRect(QtCore.QRectF())
            self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def BatchProcessing(self): 
        self.bp.exec_()
        if self.bp.returnVal==0:
            None         
        elif self.bp.returnVal==1:
            path=str(self.bp.fnameOpen)+'/'
            startNumber=self.bp.lineEdit_5.text()
            startNumber=int(startNumber)
            increment=self.bp.lineEdit_6.text()
            increment=int(increment)
            if self.bp.fileNames!=[]:
                imageList=self.bp.fileNames
            else:
                imageList=self.imageFilePaths(path)
            for fileName in imageList:
                img2=cv2.imread(fileName)
                self.imgBatch=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                if self.bp.checkBox.isChecked():
                    width=self.bp.lineEdit.text()
                    width=int(width)
                    height=self.bp.lineEdit_2.text()
                    height=int(height)
                    if self.bp.ao.checkBox_11.isChecked() and width==height:
                        TARGET_PIXEL_AREA = width*height
                        ratio = float(self.imgBatch.shape[1]) / float(self.imgBatch.shape[0])
                        new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
                        new_w = int((new_h * ratio) + 0.5)
                        self.imgBatch = cv2.resize(self.imgBatch, (new_w,new_h))
                    else:
                        self.imgBatch=cv2.resize(self.imgBatch, (width, height))
                if self.bp.checkBox_2.isChecked():
                    percent=self.bp.lineEdit_3.text()
                    percent=int(percent)
                    percent=float(percent)/100
                    self.imgBatch=cv2.resize(self.imgBatch, (0,0), fx=percent, fy=percent)
                if self.bp.ao.checkBox.isChecked():
                    self.imgBatch = cv2.flip(self.imgBatch, 0)
                if self.bp.ao.checkBox_2.isChecked():
                    self.imgBatch = cv2.flip(self.imgBatch, 1)          
                if self.bp.ao.checkBox_3.isChecked():
                    num_rows, num_cols, depth = self.imgBatch.shape
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -90, 1)
                    self.imgBatch = cv2.warpAffine(self.imgBatch, rotation_matrix, (num_cols, num_rows))
                if self.bp.ao.checkBox_4.isChecked():
                    num_rows, num_cols, depth = self.imgBatch.shape
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
                    self.imgBatch = cv2.warpAffine(self.imgBatch, rotation_matrix, (num_cols, num_rows))
                if self.bp.ao.checkBox_5.isChecked():
                    self.imgBatch=cv2.cvtColor(self.imgBatch, cv2.COLOR_RGB2GRAY)
                if self.bp.ao.checkBox_6.isChecked():
                    self.imgBatch=self.imgBatch[...,[0, 2, 1]]
                if self.bp.ao.checkBox_7.isChecked():
                    self.imgBatch=cv2.cvtColor(self.imgBatch, cv2.COLOR_RGB2BGR)
                if self.bp.ao.checkBox_8.isChecked():
                    self.imgBatch=self.imgBatch[...,[1, 2, 0]]
                if self.bp.ao.checkBox_9.isChecked():
                    self.imgBatch=self.imgBatch[...,[1, 0, 2]]
                if self.bp.ao.checkBox_10.isChecked():
                    self.imgBatch=self.imgBatch[...,[2, 0, 1]]
                if self.bp.ao.checkBox_12.isChecked():
                    im=Image.fromarray(self.imgBatch, 'RGB')
                    factor=self.bp.ao.lineEdit.text()
                    factor=int(factor)
                    enhancer=ImageEnhance.Brightness(im).enhance(factor)
                    self.imgBatch=np.array(enhancer)
                if self.bp.ao.checkBox_13.isChecked():
                    im=Image.fromarray(self.imgBatch, 'RGB')
                    factor=self.bp.ao.lineEdit_2.text()
                    factor=int(factor)
                    enhancer=ImageEnhance.Contrast(im).enhance(factor)
                    self.imgBatch=np.array(enhancer)
                if self.bp.ao.checkBox_14.isChecked():
                    self.imgBatch=(self.imgBatch)/255.0
                    power=self.bp.ao.lineEdit_3.text()
                    power=float(power)
                    self.imgBatch = cv2.pow(self.imgBatch, power)
                    self.imgBatch=np.uint8(img*255)
                if self.bp.ao.checkBox_15.isChecked():
                    sigma=self.bp.ao.lineEdit_4.text()
                    sigma=float(sigma)
                    self.imgBatch=cv2.GaussianBlur(self.imgBatch,(5,5),sigma)
                if self.bp.ao.checkBox_16.isChecked():
                    size=self.bp.ao.lineEdit_5.text()
                    size=int(size)
                    self.imgBatch=cv2.medianBlur(self.imgBatch, size)
                if self.bp.checkBox_3.isChecked() and self.bp.checkBox_4.isChecked():
                    imgNumber=''
                    rename=self.bp.lineEdit_4.text()
                    rename=str(rename)
                    numberOfDigits=len(rename)-rename.find('#')
                    firstPart=rename[0:rename.find('#')]
                    formatString=str(self.bp.comboBox.currentText())
                    totalFormat=formatString[formatString.rfind('.'):formatString.rfind(' ')]
                    for i in range(numberOfDigits-len(str(startNumber))):
                        imgNumber=imgNumber+'0'
                    imgNumber=imgNumber+str(startNumber)
                    startNumber=startNumber+increment    
                    onlyFileName=firstPart+imgNumber+totalFormat
                    fullFileName=self.bp.fnameSave+'/'+onlyFileName
                    if self.bp.ao.checkBox_5.isChecked():
                        im=Image.fromarray(self.imgBatch)
                    else:
                        im=Image.fromarray(self.imgBatch, 'RGB')
                    fullFileName=str(fullFileName)
                    im.save(fullFileName)
                    
                elif not(self.bp.checkBox_3.isChecked()) and self.bp.checkBox_4.isChecked():
                    formatString=str(self.bp.comboBox.currentText())
                    totalFormat=formatString[formatString.rfind('.'):formatString.rfind(' ')]
                    fullFileName=self.bp.fnameSave+fileName[fileName.rfind('/'):fileName.rfind('.')]+totalFormat
                    if self.bp.ao.checkBox_5.isChecked():
                        im=Image.fromarray(self.imgBatch)
                    else:
                        im=Image.fromarray(self.imgBatch, 'RGB')
                    fullFileName=str(fullFileName)
                    im.save(fullFileName)
                elif self.bp.checkBox_3.isChecked() and not(self.bp.checkBox_4.isChecked()):
                    imgNumber=''
                    rename=self.bp.lineEdit_4.text()
                    rename=str(rename)
                    numberOfDigits=len(rename)-rename.find('#')
                    firstPart=rename[0:rename.find('#')]
                    for i in range(numberOfDigits-len(str(startNumber))):
                        imgNumber=imgNumber+'0'
                    imgNumber=imgNumber+str(startNumber)
                    startNumber=startNumber+increment
                    fullFileName=self.bp.fnameSave+'/'+firstPart+imgNumber+fileName[fileName.rfind('.'):]
                    if self.bp.ao.checkBox_5.isChecked():
                        im=Image.fromarray(self.imgBatch)
                    else:
                        im=Image.fromarray(self.imgBatch, 'RGB')
                    fullFileName=str(fullFileName)
                    im.save(fullFileName)
                elif not(self.bp.checkBox_3.isChecked()) and not(self.bp.checkBox_4.isChecked()):
                    saveFileName=str(self.bp.fnameSave)
                    a=fileName.rfind('/')   
                    fileNameFull=fileName[a:]
                    fileNameFull=str(fileNameFull)
                    saveFullFileName=saveFileName+fileNameFull
                    if self.bp.ao.checkBox_5.isChecked():
                        im=Image.fromarray(self.imgBatch)
                    else:
                        im=Image.fromarray(self.imgBatch, 'RGB')
                    im.save(saveFullFileName)
    
    def contrast(self):
        global currentPath
        img_to_yuv = cv2.cvtColor(self.img,cv2.COLOR_RGB2YUV)
        img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
        self.img = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)
        self.undoStack.append(self.img)
        self.undoStackSize+=1
        self.undoStackSizeCopy=self.undoStackSize
        im=Image.fromarray(self.img, 'RGB')
        im.save(currentPath+'temporary/contrast.png')
        self.view.scene=QtGui.QGraphicsScene(self)
        self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/contrast.png'))
        self.view.setScene(self.view.scene)
        self.view.scene.setSceneRect(QtCore.QRectF())
        self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
    def simpleThresholding(self):
        global currentPath
        td=ThresholdDialog()
        td.exec_()
        if td.returnVal==0:
            None
        elif td.returnVal==1:
            choice = QtGui.QMessageBox.question(self, 'Warning', "The image will be converted to gray scale. Are you sure you want to continue?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if choice==QtGui.QMessageBox.Yes:
                tValue=td.lineEdit.text()
                tValue=float(tValue)
                self.img=cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
                ret,self.img = cv2.threshold(self.img,tValue,255,cv2.THRESH_BINARY)
                self.undoStack.append(self.img)
                self.undoStackSize+=1
                self.undoStackSizeCopy=self.undoStackSize
                im=Image.fromarray(self.img)
                im.save(currentPath+'temporary/sThresholding.png')
                self.view.scene=QtGui.QGraphicsScene(self)
                self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/sThresholding.png'))
                self.view.setScene(self.view.scene)
                self.view.scene.setSceneRect(QtCore.QRectF())
                self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            else:
                None

    def adaptiveThresholding(self):
        global currentPath
        at=AdapThreshold()
        at.exec_()
        if at.returnVal==0:
            None
        elif at.returnVal==1:
            blockSize=at.lineEdit.text()
            blockSize=int(blockSize)
            choice = QtGui.QMessageBox.question(self, 'Warning', "The image will be converted to gray scale. Are you sure you want to continue?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if choice==QtGui.QMessageBox.Yes:
                if at.checkBox.isChecked():
                    self.img=cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
                    self.img=cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, 2)
                    self.undoStack.append(self.img)
                    self.undoStackSize+=1
                    self.undoStackSizeCopy=self.undoStackSize
                    im=Image.fromarray(self.img)
                    im.save(currentPath+'temporary/aThresholding.png')
                    self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/aThresholding.png'))
                    self.view.setScene(self.view.scene)
                    self.view.scene.setSceneRect(QtCore.QRectF())
                    self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                elif at.checkBox_2.isChecked():
                    self.img=cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
                    self.img=cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, 2)
                    self.undoStack.append(self.img)
                    self.undoStackSize+=1
                    self.undoStackSizeCopy=self.undoStackSize
                    im=Image.fromarray(self.img)
                    im.save(currentPath+'temporary/aThresholding.png')
                    self.view.scene=QtGui.QGraphicsScene(self)
                    self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/aThresholding.png'))
                    self.view.setScene(self.view.scene)
                    self.view.scene.setSceneRect(QtCore.QRectF())
                    self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                else:
                    None
                    
    def otsu(self):
        global currentPath
        choice = QtGui.QMessageBox.question(self, 'Warning', "The image will be converted to gray scale. Are you sure you want to continue?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice==QtGui.QMessageBox.Yes:
            self.img=cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
            self.img = cv2.GaussianBlur(self.img,(5,5),0)
            ret3,self.img = cv2.threshold(self.img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            self.undoStack.append(self.img)
            self.undoStackSize+=1
            self.undoStackSizeCopy=self.undoStackSize
            im=Image.fromarray(self.img)
            im.save(currentPath+'temporary/otsuThresholding.png')
            self.view.scene=QtGui.QGraphicsScene(self)
            self.view.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/otsuThresholding.png'))
            self.view.setScene(self.view.scene)
            self.view.scene.setSceneRect(QtCore.QRectF())
            self.view.fitInView(self.view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            None
    
    def severityAssessment(self):
        global imgCrop
        global croppedFname
        if croppedFname=='' and self.fname=='':
            imgCrop=''
        elif croppedFname!='' and self.fname=='':
            imgCrop=''
        elif croppedFname=='' and self.fname!='':
            imgCrop=str(self.fname)

        elif croppedFname!='' and self.fname!='':
            imgCrop=croppedFname
        if imgCrop=='':
            choice = QtGui.QMessageBox.question(self, 'Important!', "Please select an image for Severity Assessment.", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if choice==QtGui.QMessageBox.Yes:
                self.showDialog()
            elif choice==QtGui.QMessageBox.No:
                None
        elif imgCrop==self.fname:
            self.cl.scene=QtGui.QGraphicsScene(self.cl)
            self.cl.scene.addPixmap(QtGui.QPixmap(imgCrop))
            self.cl.graphicsView.setScene(self.cl.scene)
            self.cl.scene.setSceneRect(QtCore.QRectF())
            self.cl.graphicsView.fitInView(self.cl.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.cl.exec_()
            self.statusbar.showMessage('Please Wait! Processing Image!')
            self.doAssessment(imgCrop)
        else:
            self.cl.scene=QtGui.QGraphicsScene(self.cl)
            self.cl.scene.addPixmap(QtGui.QPixmap(imgCrop))
            self.cl.graphicsView.setScene(self.cl.scene)
            self.cl.scene.setSceneRect(QtCore.QRectF())
            self.cl.graphicsView.fitInView(self.cl.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.cl.exec_()
            self.statusbar.showMessage('Please Wait! Processing Image!')
            self.doAssessment(imgCrop)

    def doAssessment(self, imageFileName):
    	global eseverity
    	global sseverity
    	global iseverity 
        eseverity=0
        sseverity=0
        iseverity=0
        modelErythema=self.cl.lineEdit.text()
        modelErythema=str(modelErythema)
        modelScaling=self.cl.lineEdit_2.text()
        modelScaling=str(modelScaling)
        modelInduration=self.cl.lineEdit_3.text()
        modelInduration=str(modelInduration)          
        if self.cl.returnVal==1:
            if self.cl.checkBox.isChecked():
                with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                    model1=load_model(modelErythema)
                L=model1.layers[0].input_shape
                img=cv2.imread(imageFileName)  
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img=cv2.resize(img, (L[1], L[2]))
                img=np.float32(img)
                img=np.expand_dims(img, axis=0)
                P=model1.predict(img)
                eseverity = P.argmax(axis=-1)
                eseverity=int(eseverity)
            if self.cl.checkBox_2.isChecked():
                with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                    model2=load_model(modelScaling)
                L=model2.layers[0].input_shape
                img=cv2.imread(imageFileName)  
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img=cv2.resize(img, (L[1], L[2]))
                img=np.float32(img)
                img=np.expand_dims(img, axis=0)
                P=model2.predict(img)
                sseverity = P.argmax(axis=-1)
                sseverity=int(sseverity)
            if self.cl.checkBox_3.isChecked():
                with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                    model3=load_model(modelInduration)
                L=model3.layers[0].input_shape 
                img=cv2.imread(imageFileName)  
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img=cv2.resize(img, (L[1], L[2]))
                img=np.float32(img)
                img=np.expand_dims(img, axis=0)
                P=model3.predict(img)
                iseverity = P.argmax(axis=-1)
                iseverity=int(iseverity)
            if not(self.cl.checkBox.isChecked()) and not(self.cl.checkBox_2.isChecked()) and not(self.cl.checkBox_3.isChecked()):
                choice = QtGui.QMessageBox.question(self, 'Important!', "Please select an assessment option.", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                if choice==QtGui.QMessageBox.Yes:
                    self.severityAssessment()
            self.statusbar.clearMessage()
            self.RW=ReportWindow(eseverity, sseverity, iseverity)
            self.RW.show()
        elif self.cl.returnVal==0:
            None
            

    def histopathologySegmentation(self):
    	global epiMask
    	global derMask
    	global nontMask
    	global p1
    	global currentPath
        modelSeg=load_model(currentPath+'EpidermisSegmentationModel.h5')
        img1=cv2.imread(str(self.fname))
        img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2=cv2.imread(str(self.fname))
        img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3=cv2.imread(str(self.fname))
        img3=cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        choice = QtGui.QMessageBox.question(self, 'Important!', "Please make the epidermis vertical to calculate it's maximum and minimum width. Are you sure you want to rotate to make the epidermis vertical?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        img11=img1
        img22=img2
        img33=img3
        img1=cv2.resize(img1, (1280, 960))
        img1=np.expand_dims(img1, axis=0)
        p1=modelSeg.predict(img1)
        p11=np.reshape(p1, (-1, 3))
        m1=np.argmax(p11, axis=-1)
        global d
        d=np.reshape(m1, ((p1.shape)[1], (p1.shape)[2]))
        d1=d
        d2=d
        d3=d
        [row1,col1]=np.where(d1==2)
        epiMask=np.zeros(d1.shape)
        epiMask[row1,col1]=1
        epiMask=cv2.resize(epiMask, ((img11.shape)[1], (img11.shape)[0]))
        epiMask=epiMask*255
        (thresh1, im_bw1) = cv2.threshold(epiMask, 127, 255, 0)
        im_bw1=np.uint8(im_bw1)
        im21, contours1, hierarchy1 = cv2.findContours(im_bw1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img11=cv2.drawContours(img11, contours1, -1, (0,255,0), 7)
        if choice==QtGui.QMessageBox.Yes:
        	num_rows1 = (img11.shape)[0]
        	num_cols1 = (img11.shape)[1]
        	rotation_matrix1 = cv2.getRotationMatrix2D((num_cols1/2, num_rows1/2), -90, 1)
        	img11 = cv2.warpAffine(img11, rotation_matrix1, (num_cols1, num_rows1))
        if choice==QtGui.QMessageBox.No:
        	None
        im1=Image.fromarray(img11)
        im1.save(currentPath+'temporary/epidermis.jpg')
        img2=cv2.resize(img2, (1280, 960))
        img2=np.expand_dims(img2, axis=0)
        [row2,col2]=np.where(d2==1)
        derMask=np.zeros(d2.shape)
        derMask[row2,col2]=1
        derMask=cv2.resize(derMask, ((img22.shape)[1], (img22.shape)[0]))
        derMask=derMask*255
        (thresh2, im_bw2) = cv2.threshold(derMask, 127, 255, 0)
        im_bw2=np.uint8(im_bw2)
        im22, contours2, hierarchy2 = cv2.findContours(im_bw2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img22=cv2.drawContours(img22, contours2, -1, (0,255,0), 7)
        if choice==QtGui.QMessageBox.Yes:
        	num_rows2 = (img22.shape)[0]
        	num_cols2 = (img22.shape)[1]
        	rotation_matrix2 = cv2.getRotationMatrix2D((num_cols2/2, num_rows2/2), -90, 1)
        	img22 = cv2.warpAffine(img22, rotation_matrix2, (num_cols2, num_rows2))
        if choice==QtGui.QMessageBox.No:
        	None
        im2=Image.fromarray(img22)
        im2.save(currentPath+'temporary/dermis.jpg')
        img3=cv2.resize(img3, (1280, 960))
        img3=np.expand_dims(img3, axis=0)
        [row3,col3]=np.where(d3==0)
        nontMask=np.zeros(d3.shape)
        nontMask[row3,col3]=1
        nontMask=cv2.resize(nontMask, ((img33.shape)[1], (img33.shape)[0]))
        nontMask=nontMask*255
        (thresh3, im_bw3) = cv2.threshold(nontMask, 127, 255, 0)
        im_bw3=np.uint8(im_bw3)
        im23, contours3, hierarchy3 = cv2.findContours(im_bw3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img33=cv2.drawContours(img33, contours3, -1, (0,255,0), 7)
        if choice==QtGui.QMessageBox.Yes:
        	num_rows3 = (img33.shape)[0]
        	num_cols3 = (img33.shape)[1]
        	rotation_matrix3 = cv2.getRotationMatrix2D((num_cols3/2, num_rows3/2), -90, 1)
        	img33 = cv2.warpAffine(img33, rotation_matrix3, (num_cols3, num_rows3))
        if choice==QtGui.QMessageBox.No:
        	None
        im3=Image.fromarray(img33)
        im3.save(currentPath+'temporary/nontissue.jpg')
        global SW
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/epidermis.jpg'))
        SW.graphicsView.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/dermis.jpg'))
        SW.graphicsView_2.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView_2.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        SW.scene=QtGui.QGraphicsScene()
        SW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/nontissue.jpg'))
        SW.graphicsView_3.setScene(SW.scene)
        SW.scene.setSceneRect(QtCore.QRectF())
        SW.graphicsView_3.fitInView(SW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        if choice==QtGui.QMessageBox.Yes:
        	num_rows4 = (im_bw1.shape)[0]
        	num_cols4 = (im_bw1.shape)[1]
        	rotation_matrix4 = cv2.getRotationMatrix2D((num_cols4/2, num_rows4/2), -90, 1)
        	im_bw1 = cv2.warpAffine(im_bw1, rotation_matrix4, (num_cols4, num_rows4))
        if choice==QtGui.QMessageBox.No:
        	None
        SW.show()

    def crop(self):
        fname=QtGui.QFileDialog.getSaveFileName(None, 'SaveFile','',"Image file(*.png *.jpg *.bmp *.jpeg)")
        global croppedFname
        if imgCrop==self.fname and croppedFname=='':
            if (len(self.img.shape))==3:
                im=Image.fromarray(self.img, 'RGB')
            else:
                im=Image.fromarray(self.img)
            im.save(str(fname))
        elif croppedFname!='':
            im=cv2.imread(croppedFname)
            if (len(im.shape))==3:
                im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im=Image.fromarray(im, 'RGB')
            else:
                im=Image.fromarray(im)
            im.save(str(fname))

		   
    def mm(self):
		global currentPath
		c=str(self.fname)
		img=cv2.imread(c)
		img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		segments=slic(img, n_segments=550, compactness=40, convert2lab=True, max_iter=10, sigma=0)
		centro=[]
		scmodel=load_model(currentPath+'StratumCorneumSegmentation.h5')
		img=cv2.imread(c)
		img6=cv2.imread(c)
		img6=cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)
		img1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img1=img1/255.0
		img1=cv2.resize(img1, (1280, 960))
		img1=np.expand_dims(img1, axis=0)
		p=scmodel.predict(img1)
		p=np.reshape(p, (960, 1280))
		[row, col]=np.where(p>0.5)
		Mask=np.zeros(((img1.shape)[1], (img1.shape)[2]))
		Mask[row, col]=1
		Mask=cv2.resize(Mask, ((img.shape)[1], (img.shape)[0]))
		Mask=Mask*255
		_, Mask=cv2.threshold(Mask, 127, 255, 0)
		regions = regionprops(segments)
		Mask=np.int64(Mask)
		for props in regions:
			x, y = props.centroid
			x=np.int64(x)
			y=np.int64(y)
			centro.append([x, y])
		centroFinal=[]
		for i in centro:
			if Mask[i[0], i[1]]==0:	
				centroFinal.append(i)
		neutroModel=load_model(currentPath+'neutroPredModelFinal.h5', custom_objects={'CapsuleLayer':CapsuleLayer, 'Length':Length, 'TrimmedAveragePool':TrimmedAveragePool})
		croppedImg=[]
		probValues=[]
		for i in range(len(centroFinal)):
			j=centroFinal[i]
			if j[0]<111:
				x1=0
				x2=224
			else:
				x1=j[0]-111
				x2=j[0]+112
			if j[1]<111:
				y1=0
				y2=224
			else:
				y1=j[1]-111
				y2=j[1]+112
			cropped_img=img6[x1:x2, y1:y2, :]
			cropped_img=np.float64(cropped_img)
			cropped_img=cropped_img/255.0
			cropped_img=cv2.resize(cropped_img, (224, 224))
			cropped_img=np.expand_dims(cropped_img, axis=0)
			p=neutroModel.predict(cropped_img)
			if p>=0.8511:
				probValues.append(p)
				croppedImg.append([x1, y1, x2, y2, p])
		croppedImg.sort(key=takeForth, reverse=True)	
		reg=len(croppedImg)
		MMDialog=munroMicroabscessDialog(reg)
		MMDialog.exec_()
		image=cv2.imread(c)
		image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if MMDialog.returnVal==1:
			if MMDialog.checkBox.isChecked():
				l=len(croppedImg)
				if l<5:
					last=l
				else:
					last=5
				for i in range(last):
					y1=(croppedImg[i])[1]
					x1=(croppedImg[i])[0]
					y2=(croppedImg[i])[3]
					x2=(croppedImg[i])[2]
					image=cv2.rectangle(image, (y1, x1), (y2, x2), (0,255,0), 3)
				im=Image.fromarray(image)
				im.save(currentPath+'temporary/top5.jpg')
				self.img=image
				setSceneGraphicsView(self.view, currentPath+'temporary/top5.jpg')
			elif MMDialog.checkBox_2.isChecked():
				l=len(croppedImg)
				if l<10:
					last=l
				else:
					last=10
				for i in range(last):
					image=cv2.rectangle(image, ((croppedImg[i])[1], (croppedImg[i])[0]), ((croppedImg[i])[3], (croppedImg[i])[2]), (0,255,0), 3)
				im=Image.fromarray(image)
				im.save(currentPath+'temporary/top10.jpg')
				self.img=image
				setSceneGraphicsView(self.view, currentPath+'temporary/top10.jpg')
			elif MMDialog.checkBox_3.isChecked():
				l=len(croppedImg)
				if l<15:
					last=l
				else:
					last=15
				for i in range(last):
					image=cv2.rectangle(image, ((croppedImg[i])[1], (croppedImg[i])[0]), ((croppedImg[i])[3], (croppedImg[i])[2]), (0,255,0), 3)
				im=Image.fromarray(image)
				im.save(currentPath+'temporary/top15.jpg')
				self.img=image
				setSceneGraphicsView(self.view, currentPath+'temporary/top15.jpg')
			elif MMDialog.checkBox_4.isChecked():
				l=len(croppedImg)
				if l<20:
					last=l
				else:
					last=20
				for i in range(last):
					image=cv2.rectangle(image, ((croppedImg[i])[1], (croppedImg[i])[0]), ((croppedImg[i])[3], (croppedImg[i])[2]), (0,255,0), 3)
				im=Image.fromarray(image)
				im.save(currentPath+'temporary/top20.jpg')
				self.img=image
				setSceneGraphicsView(self.view, currentPath+'temporary/top20.jpg')
			elif MMDialog.checkBox_5.isChecked():
				l=len(croppedImg)
				if l<25:
					last=l
				else:
					last=25
				for i in range(last):
					image=cv2.rectangle(image, ((croppedImg[i])[1], (croppedImg[i])[0]), ((croppedImg[i])[3], (croppedImg[i])[2]), (0,255,0), 3)
				im=Image.fromarray(image)
				im.save(currentPath+'temporary/top25.jpg')
				self.img=image
				setSceneGraphicsView(self.view, currentPath+'temporary/top25.jpg')
			elif MMDialog.checkBox_6.isChecked():
				l=len(croppedImg)
				for i in range(l):
					image=cv2.rectangle(image, ((croppedImg[i])[1], (croppedImg[i])[0]), ((croppedImg[i])[3], (croppedImg[i])[2]), (0,255,0), 3)
				im=Image.fromarray(image)
				im.save(currentPath+'temporary/showAll.jpg')
				self.img=image
				setSceneGraphicsView(self.view, currentPath+'temporary/showAll.jpg')
			else:
				None
		else:
			None
		row=Mask[0,:]
		global lengthSCMax
		global lengthSCMin
		lengthSCMax=0
		lengthSCMin=(Mask.shape)[1]
		for i in range((Mask.shape)[0]):
			row=Mask[i,:]
			index=np.argwhere(row==0)
			c=index.shape
			if c[0]!=0:
				left=int(index[0])
				right=int(index[(c[0]-1)])
				for i in range(c[0]-1):
				    if (index[i+1]-index[i])>300:
				        left=index[i+1]
				j=c[0]-1
				while j>=0:
				    if (index[j]-index[j-1])>300:
				        right=index[j-1]
				    j=j-1
				length=right-left
			else:
				continue
			if length>50:
				if lengthSCMax<length:
					lengthSCMax=length
				if lengthSCMin>length:
					lengthSCMin=length
		lengthSCMax=int(lengthSCMax)
		lengthSCMin=int(lengthSCMin)
		if MMDialog.checkBox_7.isChecked():
			self.USCW=UniformitySCReport(lengthSCMax, lengthSCMin)
			self.USCW.show()
			
    def diseaseDetection(self):
        global imgCrop
        global croppedFname
        if croppedFname=='' and self.fname=='':
            imgCrop=''
        elif croppedFname!='' and self.fname=='':
            imgCrop=''
        elif croppedFname=='' and self.fname!='':
            imgCrop=str(self.fname)

        elif croppedFname!='' and self.fname!='':
            imgCrop=croppedFname
        if imgCrop=='':
            choice = QtGui.QMessageBox.question(self, 'Important!', "Please select an image for Disease Detection.", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if choice==QtGui.QMessageBox.Yes:
                self.showDialog()
            elif choice==QtGui.QMessageBox.No:
                None
        elif imgCrop==self.fname:
            self.statusbar.showMessage('Please Wait! Processing Image!')
            self.doDetection(imgCrop)
        else:
            self.statusbar.showMessage('Please Wait! Processing Image!')
            self.doDetection(imgCrop)
			
    def doDetection(self, img1Crop):
        global currentPath
        global DisMask
        model=load_model(currentPath+'DiseasedPredict.h5')
        img=cv2.imread(str(img1Crop))			
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2=img
        img=cv2.resize(img, (256, 256))
        img=np.expand_dims(img, axis=0)
        pred=model.predict(img)
        pred=np.reshape(pred, (256, 256))
        [row, col]=np.where(pred>0.5)
        DisMask=np.zeros((256, 256))
        DisMask[row, col]=1
        DisMask=cv2.resize(DisMask, ((img2.shape)[1], (img2.shape)[0]))
        DisMask=DisMask*255
        _, DisMask=cv2.threshold(DisMask, 127, 255, 0)
        DisMask=np.uint8(DisMask)	
        im21, contours1, hierarchy1 = cv2.findContours(DisMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img2=cv2.drawContours(img2, contours1, -1, (0,255,0), 2)
        self.img=img2
        im=Image.fromarray(img2)
        im.save(currentPath+'temporary/diseasedDetectedFromModel.jpg')
        global DW
        DW.scene=QtGui.QGraphicsScene()
        DW.scene.addPixmap(QtGui.QPixmap(currentPath+'temporary/diseasedDetectedFromModel.jpg'))
        DW.graphicsView.setScene(DW.scene)
        DW.scene.setSceneRect(QtCore.QRectF())
        DW.graphicsView.fitInView(DW.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        DW.show()
         	                   
    

    def retranslateUi(self):
        self.setWindowTitle(_translate("Window", "Psoriasis Assessment", None))
        self.menuFile.setTitle(_translate("Window", "File", None))
        self.menuEdit.setTitle(_translate("Window", "Edit", None))
        self.menuApplication.setTitle(_translate("Window", "Application", None))
        self.menuFiltering.setTitle(_translate("Window", "Filtering", None))
        self.menuTransform.setTitle(_translate("Window", "Transform", None))
        self.menuFlip.setTitle(_translate("Window", "Flip", None))
        self.menuRotate.setTitle(_translate("Window", "Rotate", None))
        self.actionOpen.setText(_translate("Window", "Open", None))
        self.actionHistogram.setText(_translate("Window", "Histogram", None))
        self.actionImage_Information.setText(_translate("Window", "Image Information", None))
        self.actionBatch_Processing.setText(_translate("Window", "Batch Processing", None))
        self.actionSave_As.setText(_translate("Window", "Save As", None))
        self.actionExit.setText(_translate("Window", "Exit", None))
        self.actionUndo.setText(_translate("Window", "Undo", None))
        self.actionRedo.setText(_translate("Window", "Redo", None))
        self.actionCrop.setText(_translate("Window", "Crop current selection", None))
        self.actionColor_channel_decomposition.setText(_translate("Window", "Color channel decomposition", None))
        self.actionMean_Filtering.setText(_translate("Window", "Mean Filtering", None))
        self.actionMaximum_Filtering.setText(_translate("Window", "Maximum Filtering", None))
        self.actionMedian_Filtering.setText(_translate("Window", "Median Filtering", None))
        self.actionContrast.setText(_translate("Window", "Contrast", None))
        self.menuThresholding.setTitle(_translate("Window", "Thresholding", None))
        self.actionRight.setText(_translate("Window", "Horizontal", None))
        self.actionLeft.setText(_translate("Window", "Vertical", None))
        self.actionRight_2.setText(_translate("Window", "Right", None))
        self.actionLeft_2.setText(_translate("Window", "Left", None))
        self.actionSimpleThresholding.setText(_translate("Window", "Simple Thresholding", None))
        self.actionOtsu.setText(_translate("Window", "Otsu", None))
        self.actionAdaptiveThresholding.setText(_translate("Window", "Adaptive Thresholding", None))
        self.actionSeverityAssessment.setText(_translate("Window", "Severity Assessment", None))
        self.actionDiseaseDetection.setText(_translate("Window", "Diseased Region Detection", None))
        self.actionHistopathologySegmentation.setText(_translate("Window", "Histopathology Segmentation", None))
        self.actionMunrosMicroabcess.setText(_translate("Window", "Munro's microabscess", None))

def takeForth(elem):
	return elem[4]
	
def setSceneGraphicsView(view, fname):
	view.scene=QtGui.QGraphicsScene()
	view.scene.addPixmap(QtGui.QPixmap(fname))
	view.setScene(view.scene)
	view.scene.setSceneRect(QtCore.QRectF())
	view.fitInView(view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    ui = Window()
    sys.exit(app.exec_())
