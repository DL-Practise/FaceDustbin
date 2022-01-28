# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import copy
import xml.etree.cElementTree as et
import os
import cv2
import math
import time
from PIL import Image
import numpy as np
import threading
import json
import RPi.GPIO as GPIO


# ui配置文件
cUi, cBase = uic.loadUiType("image_widget.ui")

# 主界面
class ImageWidget(QWidget, cUi):
    action_sig = pyqtSignal(int)
    
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        
        with open('./cfg.json', 'r') as f:
            map_info = json.load(f)
            self.face_rect = map_info['face_rect']
            self.name = map_info['name']
        
        self.sliderWidth.setValue(int(self.face_rect[2] - self.face_rect[0]))
        self.sliderHeight.setValue(self.face_rect[3] - self.face_rect[1])

        self.qpixmap = None
        self.info = None
        
        self.camera_cap = None
        self.capture_seq = 0  
        self.capture_dir = './data/0/'
        self.train_dir = './data/'
        #LBPHFaceRecognizer/EigenFaceRecognizer/FisherFaceRecognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create() 
        if os.path.exists('model/model.yml'):
            self.face_recognizer.read('model/model.yml')


        self.thread_capture = {'flag': False, 
                               'tag': 'capture',
                               'handle': None}
        self.thread_train = {'flag': False, 
                             'tag': 'train',
                             'handle': None}
        self.thread_infer = {'flag': False, 
                             'tag': 'infer',
                             'handle': None}

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(40, GPIO.OUT, initial=False)
        self.pwm = GPIO.PWM(40, 50)
        self.pwm.start(0)
        self.pwm.ChangeDutyCycle(5)
        time.sleep(0.2)
        self.pwm.ChangeDutyCycle(0)
        self.timer = QTimer()
        self.action = False
        
        self.action_sig.connect(self.action_slot)
                            
    @pyqtSlot()    
    def on_btnCamera_clicked(self):
        print('on_btnCamera_clicked')
        self.thread_infer['flag'] = True
        self.thread_infer['handle'] = threading.Thread(target=self.thread_infer_func, args=())
        self.thread_infer['handle'].start()
                    
    @pyqtSlot()    
    def on_btnStop_clicked(self):
        print('on_btnStop_clicked')
        self.thread_capture['flag'] = False
        self.thread_train['flag'] = False
        self.thread_infer['flag'] = False
        
        if self.thread_capture['handle'] is not None:
            self.thread_capture['handle'].join()
            self.thread_capture['handle'] = None
            
        if self.thread_train['handle'] is not None:
            self.thread_train['handle'].join()
            self.thread_train['handle'] = None
            
        if self.thread_infer['handle'] is not None:
            self.thread_infer['handle'].join()
            self.thread_infer['handle'] = None

        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None
            
        self.pwm.stop()
        GPIO.cleanup()
   
    @pyqtSlot()    
    def on_btnCpature_clicked(self):
        print('on_btnCpature_clicked')
        self.thread_capture['flag'] = True
        self.thread_capture['handle'] = threading.Thread(target=self.thread_capture_func, args=())
        self.thread_capture['handle'].start()
        
    @pyqtSlot()    
    def on_btnDir_clicked(self):
        print('on_btnDir_clicked')
        capture_dir = os.path.abspath(self.capture_dir)
        sys_op = sys.platform
        if 'win' in sys_op or 'WIN' in sys_op or 'Win' in sys_op:
            os.startfile(capture_dir)
        else:
            import subprocess
            opener ="open" if sys.platform == "darwin" else "xdg-open" 
            subprocess.call([opener, capture_dir]) 
        
    @pyqtSlot()    
    def on_btnTrain_clicked(self):
        print('on_btnTrain_clicked')
        self.thread_train['flag'] = True
        self.thread_train['handle'] = threading.Thread(target=self.thread_train_func, args=())
        self.thread_train['handle'].start()
        
    @pyqtSlot()    
    def on_btnKai_clicked(self):
        print('on_btnKai_clicked')
        self.pwm.ChangeDutyCycle(10)
        time.sleep(0.2)
        self.pwm.ChangeDutyCycle(0)
    
    @pyqtSlot()    
    def on_btnGuan_clicked(self):
        print('on_btnGuan_clicked')
        self.pwm.ChangeDutyCycle(5)
        time.sleep(0.2)
        self.pwm.ChangeDutyCycle(0)
        
    def on_sliderWidth_valueChanged(self):
        value = self.sliderWidth.value()
        print('now slider width value is:', value)    
        x_center = (self.face_rect[2] + self.face_rect[0]) / 2.0
        self.face_rect[0] = int(x_center - value / 2.0)
        self.face_rect[2] = int(x_center + value / 2.0)
        
    def on_sliderHeight_valueChanged(self):
        value = self.sliderHeight.value()
        print('now slider height value is:', value)    
        y_center = (self.face_rect[3] + self.face_rect[1]) / 2.0
        self.face_rect[1] = int(y_center - value / 2.0)
        self.face_rect[3] = int(y_center + value / 2.0)
        
        
    def thread_capture_func(self):
        print('thread_capture_func start...')
        
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
            
        self.camera_cap = cv2.VideoCapture(int(0))
        fream_seq = 1
        capture_seq = 1
        while self.thread_capture['flag'] is True:
            ret, img = self.camera_cap.read()
            if ret is False:
                print('error: read camera frame failed')
                #self.stop_all()
                return
            img = cv2.resize(img, (640, 480))
            img = cv2.flip(img, 1) 
            (x, y, w, h) = self.face_rect
            cv2.rectangle(img, (self.face_rect[0]-5, self.face_rect[1]-5), (self.face_rect[2]+5, self.face_rect[3]+5), (0, 255, 0), 2)
            
            if fream_seq % 30 == 0:
                img_save = img[self.face_rect[1]:self.face_rect[3],self.face_rect[0]:self.face_rect[2],:] 
                save_file = os.path.join(self.capture_dir, '%f.jpg'%time.time())
                cv2.imwrite(save_file, img_save)
                capture_seq += 1
               
            # for show img
            height, width, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * width
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.qpixmap = QPixmap.fromImage(image) 
            # for show text
            self.info = {'tag': '抓取第%d张图片!'%(capture_seq),
                         'pos': [100, 100],
                         'color': QColor(255, 0, 0),
                         'size': 4}  

            self.update()
            fream_seq += 1

    def thread_infer_func(self):
        print('thread_infer_func start...')
        self.camera_cap = cv2.VideoCapture(int(0))
        fream_seq = 1
        while self.thread_infer['flag'] is True:
            ret, img = self.camera_cap.read()
            if ret is False:
                print('error: read camera frame failed')
                #self.stop_all()
                return
            img = cv2.resize(img, (640, 480))
            img = cv2.flip(img, 1) 
            (x, y, w, h) = self.face_rect
            cv2.rectangle(img, (self.face_rect[0]-5, self.face_rect[1]-5), (self.face_rect[2]+5, self.face_rect[3]+5), (0, 255, 0), 2)
            
            # for recognize
            img_face = img[self.face_rect[1]:self.face_rect[3],self.face_rect[0]:self.face_rect[2],:] 
            img_face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            label, confidence = self.face_recognizer.predict(img_face_gray)
            confidence = 100 - confidence
   
            # for show img
            height, width, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * width
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.qpixmap = QPixmap.fromImage(image) 
            # for show text
            final_name = self.lineEdit.text()
            if label == 0:
                final_score = max(0, confidence)
                if final_score >= 60 and self.action is False:
                    self.action_sig.emit(1)
            else:
                final_score = 0
                
            self.info = {'tag': '%s:%d'%(final_name, final_score),
                         'pos': [int(self.width()*0.4), int(self.height()/2.0)],
                         'color': QColor(255, 0, 0),
                         'size': 4}  

            self.update()
            fream_seq += 1
        
    def thread_train_func(self):
        print('thread_train_func start...')
        dirs = os.listdir(self.train_dir)
        faces = []
        labels = []
        
        #let's go through each directory and read images within it
        for dir_name in dirs:
            label = int(dir_name)
            subject_dir_path = self.train_dir + "/" + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            print('***', label, subject_dir_path)
            for image_name in subject_images_names:
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces.append(image_gray)
                labels.append(label)
                
                # for show img
                height, width, bytesPerComponent = image.shape
                bytesPerLine = bytesPerComponent * width
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.qpixmap = QPixmap.fromImage(image) 
                # for show text
                self.info = {'tag': '正在训练',
                             'pos': [100, 100],
                             'color': QColor(255, 0, 0),
                             'size': 4}  
                self.update()

        self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.save('model/model.yml')
        
        self.thread_train['flag'] = False
        self.thread_train['handle'] = None
        
        self.info = {'tag': '完成训练',
                             'pos': [100, 100],
                             'color': QColor(255, 0, 0),
                             'size': 4}  
        self.update()
    

    def draw_image(self, painter):
        pen = QPen()
        font = QFont("Microsoft YaHei")
        if self.qpixmap is not None:
            painter.drawPixmap(QtCore.QRect(0, 0, self.width(), self.height()), self.qpixmap)
        else:
            pen.setColor(QColor(0, 0, 0))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(0, 0, self.width(), self.height())
            
    def draw_info(self, painter):
        if self.info is None:
            return
        tag = self.info['tag']
        pos = self.info['pos']
        color = self.info['color']
        size = self.info['size']

        font = QFont("宋体")
        pointsize = font.pointSize()
        font.setPixelSize(pointsize*200/72)
        painter.setFont(font)
 
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(color)
        painter.setPen(pen)
        painter.drawText(pos[0], pos[1], tag) 
        if self.action:
            painter.drawText(pos[0], pos[1]+50, 'Open Dustbin!!!!')

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.draw_image(painter)
        self.draw_info(painter)
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
            '本程序',
            "是否要退出程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            self.on_btnStop_clicked()
            with open('./cfg.json', 'w') as f:
                map_info = {'face_rect':self.face_rect,
                            'name': self.lineEdit.text()}
                json.dump(map_info, f)
        else:
            event.ignore()
   

    def action_slot(self, action):
        print('start_action')
        self.action = True
        self.pwm.ChangeDutyCycle(10)
        time.sleep(0.2)
        self.pwm.ChangeDutyCycle(0)
        self.timer.start()
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.stop_action)
            
    def stop_action(self):
        print('stop_action')
        self.timer.stop()
        self.pwm.ChangeDutyCycle(5)
        time.sleep(0.2)
        self.pwm.ChangeDutyCycle(0)
        self.action = False

if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cImageWidget = ImageWidget()
    cImageWidget.show()
    sys.exit(cApp.exec_())
