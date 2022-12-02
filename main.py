########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from dataclasses import dataclass
from multiprocessing import Process, Pipe, Array, Lock, freeze_support
from multiprocessing.shared_memory import SharedMemory
from threading import Thread, Event
import subprocess
import time
import os
import sys
import win32.win32gui as win32gui
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import cv2
import mediapipe as mp
import numpy as np
# from scipy import ndimage
from PyQt5 import QtCore, QtGui, QtWidgets

from camera import create_live_camera, SM_IMAGE_NAME, DEVICES
from frame_processor import create_frame_processor
from renderer import create_image_renderer

DEBUG_CAM_NAME = "Microsoft® LifeCam Cinema(TM)"

class MainProcessManager:
    def _print(self, string):
        self.debug_print_lock.acquire()
        try:
            print("{0: <16}: ".format("MainProcessManager") + str(string), flush=True)
        finally:
            self.debug_print_lock.release()

    def __init__(self, debug_print=False):
        self.debug_print_lock = Lock()
        self.current_source = -1
        self.image_lock = None
        self.cam_process = None
        self.FP_process = None
        self.IR_process = None
        
        self.IR_FPtoIR = None
        self.debug_print = debug_print
        
        camselect = CameraPreviewManager()
        self.cam, self.res = camselect.wait_for()
        
        self.running = True
        self.thread = Thread(name='MainProcess_main', target=self._main_loop, daemon=True)
        self.thread.start()
        
        # self.settings_thread = Thread(name='MainProcessManager_camera_settings', target=self._settings_preview_thread, daemon=True)
        # self.thread.start()
        
    def _main_loop(self):
        print("Loading. Please wait...")
        self.change_cam_source(self.cam)
        
        self.close_event.wait()
        self.close_current()
        print("Exited")
    
            
    def close_current(self):
        if self.IR_process is not None: self.IR_run_event.clear()
        if self.FP_process is not None: self.FP_run_event.clear()
        if self.cam_process is not None: self.cam_run_event.clear()
            
        if self.IR_process is not None:
            self.IR_process.join()
            print("Closed Image Renderer")
        if self.FP_process is not None:
            self.FP_process.join()
            print("Closed Frame Processor")
        if self.cam_process is not None:
            self.cam_process.join()
            print("Closed Camera")
            
        del self.IR_FPtoIR
            
    def change_cam_source(self, source):
        if isinstance(source, str): source = DEVICES.index(source)
        
        if source != self.current_source:
            
            self.close_current()
            
            success, self.cam_process, self.image_lock, self.cam_run_event, (self.FP_consumer, self.IR_consumer), self.img_shape = create_live_camera(self.debug_print_lock, 2, source=source,target_size=self.res, debug_print=self.debug_print, crop_aspect=8/9)
            
            if not success:
                if self.debug_print: self._print(f"Camera ID {source-1} could not be opened. Recreating existing camera")
                source = self.current_source
                self.current_source = -1
                self.change_cam_source(source)
                return 
            else:
                if self.debug_print: self._print(f"Camera successfully changed to ID {source-1}")
                self.current_source = source
                
                self.FP_process, self.FP_run_event, self.IR_FPtoIR = create_frame_processor(self.image_lock,self.debug_print_lock,self.FP_consumer,self.img_shape)
                
                self.IR_process, self.IR_run_event, self.close_event = create_image_renderer(self.image_lock, self.debug_print_lock, self.IR_consumer, self.IR_FPtoIR, self.img_shape)
                
                if self.debug_print: self._print(f"Sending start signal to Camera process")
                self.cam_run_event.set()
                
                if self.debug_print: self._print(f"Sending start signal to Image Renderer process")
                self.IR_run_event.set()
                
                if self.debug_print: self._print(f"Sending start signal to Frame Processor process")
                self.FP_run_event.set()
                
                
class CameraSelectDialog(QtWidgets.QDialog):
    def __init__(self, main_obj, devices_dict, parent=None):
        super(CameraSelectDialog, self).__init__(parent)
        self.main_obj = main_obj
        self.devices_dict = devices_dict
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint,False)
        if devices_dict is None:
            if len(DEVICES)==0:
                label = QtWidgets.QLabel("No supported devices found!")
                label.setStyleSheet('color: red')
                label.setFont(QtGui.QFont('Arial', 24))
                label.setAlignment(QtCore.Qt.AlignCenter)
            else:
                label = QtWidgets.QLabel("Camera devices detected, but none support the correct resolutions and/or encoding!")
                label.setStyleSheet('color: red')
                label.setFont(QtGui.QFont('Arial', 24))
                label.setAlignment(QtCore.Qt.AlignCenter)
                
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(label)
            self.setWindowTitle("Error")
        else:
            okay_button = QtWidgets.QPushButton('OK')
            okay_button.clicked.connect(self.on_clicked)
            okay_button.setFont(QtGui.QFont('Arial', 16))
            self.resbox = QtWidgets.QComboBox()
            self.resbox.setFont(QtGui.QFont('Arial', 16))
            self.resbox.currentTextChanged.connect(self.change_res_selection)
            self.combobox = QtWidgets.QComboBox()
            self.combobox.setFont(QtGui.QFont('Arial', 16))
            self.combobox.currentTextChanged.connect(self.change_selection)
            label = QtWidgets.QLabel("Select camera and resolution")
            label.setFont(QtGui.QFont('Arial', 16))
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setWordWrap(True)
            for device in devices_dict.keys():
                self.combobox.addItem(device)
            
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(label)
            lay.addWidget(self.combobox)
            lay.addWidget(self.resbox)
            lay.addWidget(okay_button)
            self.setWindowTitle("Select Camera")
    
    def change_res_selection(self, s):
        if s:
            li = s.split(" ")
            self.main_obj.res = (int(li[0]), int(li[2]))
    
    def change_selection(self, s):
        self.val = str(s)
        self.resbox.clear()
        for size in self.devices_dict[self.val]:
            self.resbox.addItem(size)
        
        self.main_obj.change_preview_camera(self.val)
        
    
    @QtCore.pyqtSlot()
    def on_clicked(self):
        # if self.main_obj.check_settings_closed():
        self.accept()
        # else:
            # self.main_obj.focus_dialog()

class CameraPreviewManager:
    def __init__(self):
        
        self.devices_dict = self._determine_suitable_devices()
        self.current_selection = None
        self.res = (640, 480)
        
        self._temp_cam = None
        self.chosen_camera = None
        self.preview_thread = None
        self.done_event = Event()
        # self.settings_thread = None
        app = QtWidgets.QApplication([sys.argv[0]])
        ex = CameraSelectDialog(self, self.devices_dict)
        ex.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # time.sleep(1)
        # ex.activateWindow()
        # ex.raise_()
        if ex.exec_() == QtWidgets.QDialog.Accepted:
            self.chosen_camera = ex.val
        else:
            print("Exiting...")
            app.quit()
            exit()
        app.quit()
        self.done_event.set()
        
    def _determine_suitable_devices(self):
        devices = {}
        try:
            for device in DEVICES:
                # print(device)
                supported_resolutions = self._get_supported_resolutions(device)
                if supported_resolutions is not None:
                    devices[device] = supported_resolutions
        except: # Very bad practice
            return None
            
        if len(devices)==0:
            return None
        else:
            return devices
        
    def _get_supported_resolutions(self, device_name):
        # Only present resolutions that can run at 30
        out = subprocess.run(f'res/ffmpeg.exe -f dshow -list_options true -i video="{device_name}"',capture_output=True)
        sizes = set()
        string = out.stderr.decode() # FFMPEG uses stderr
        string = string[string.find("DirectShow video device options"):]
        spl    = string.splitlines()[2:-1]
        for sp in spl:
            sp = sp[sp.find("max"):]
            # print(sp)
            end = sp.find(" ",sp.find("fps="))
            if end == -1: end = len(sp)
            fps = float(sp[sp.find("fps=")+4:len(sp)+1+sp.find(" ",sp.find("fps="))])
            if fps<30:
                continue
            size = sp[sp.find("s=")+2:sp.find(" ",sp.find("s="))].replace("x", " x ")
            
            li = size.split(" ")
            if min(int(li[0]),int(li[2]))>=320:
                sizes.add(size)
                # continue
                
        sizes = sorted(list(sizes),key=lambda x: int(x.split()[0]), reverse=True)
        
        if len(sizes)==0:
            return None
        else:
            return sizes
        # self.resbox.clear()
        # for size in sizes:
            # self.resbox.addItem(size)

    def change_preview_camera(self, new):
        self.current_selection = new
        # print(new)
        if self.preview_thread is not None:
            self.run_settings = False
            self.preview_thread.join()
        # if self.settings_sub is not None and self.settings_sub.poll() is None:
            # self.settings_sub.kill()
        
        self._temp_cam = cv2.VideoCapture(DEVICES.index(new), cv2.CAP_DSHOW)
        
        self.run_settings = True
        self.preview_thread = Thread(name='MainProcess_camera_preview', target=self._settings_preview_loop, daemon=True)
        self.preview_thread.start()
        
        
    def _settings_preview_loop(self):
        # ret,frame = self._temp_cam.read()
        # if ret:
            # cv2.imshow("Adjust Settings and Close Dialog Window",frame)
        # cv2.waitKey(1)
        
        while self.run_settings:
            ret,frame = self._temp_cam.read()
            if ret:
                cv2.imshow("Adjust Settings and Close Dialog Window",frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        self._temp_cam.release()
        self._temp_cam = None
        
    def wait_for(self):
        self.done_event.wait()
        if self.preview_thread is not None:
            self.run_settings = False
            self.preview_thread.join()
        # if self.settings_sub is not None and self.settings_sub.poll() is None:
            # self.settings_sub.kill()
            
        return self.current_selection, self.res

if __name__=="__main__":
    freeze_support()
    M = MainProcessManager()
    M.thread.join()
