########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from multiprocessing import current_process, Event, Process, Pipe
from multiprocessing.shared_memory import SharedMemory
import time
import os
# import json
# from itertools import filterfalse

import psutil
import numpy as np
import mediapipe as mp
import cv2
# from skimage.transform import PiecewiseAffineTransform, warp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import geometric_slerp

from camera import SM_IMAGE_NAME, SM_TIME_NAME
from packets import FPtoIR_Packet

ENABLE_OEF      = True

CONS = np.array([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                (29, 31), (30, 32), (27, 31), (28, 32)])
                
CON_DUPLEX = np.array([
                [4 , 1], # 0
                [2 , 0], # 1
                [3 , 1], # 2
                [7 , 2],# 3
                [0 , 5], #4
                [6 , 4], #5
                [5 , 8], #6
                [9 , 3], #7
                [6 ,10], #8
                [10 ,0], #9
                [0 , 9], #10
                [23,12], #11
                [11,24], #12
                [11,15], #13
                [16,12], #14
                [19,17], #15
                [18,20], #16
                [15,19], #17
                [20,16], #18
                [17,15], #19
                [16,18], #20
                [19,15], #21
                [16,20], #22
                [24,11], #23
                [12,23], #24
                [27,23], #25
                [24,28], #26
                [31,29], #27
                [30,32], #28
                [27,31], #29
                [32,30], #30
                [29,27], #31
                [28,30], #32
                ]).T

def create_frame_processor(image_lock, debug_lock, new_frame_event, img_shape, **kwargs):
    
    if "FrameProcessor" in [p.name for p in psutil.process_iter()]:
        raise RuntimeError("FrameProcessor instance already running")
    
    run_event = Event()
    IR_FPtoIR, FP_FPtoIR = Pipe(True)
    
    process = Process(target=_FrameProcessor.process_start_point, args=[image_lock, debug_lock, new_frame_event, FP_FPtoIR, img_shape, run_event],kwargs=kwargs, name="FrameProcessor", daemon=True)
    
    process.start()
    
    return process, run_event, IR_FPtoIR


class _FrameProcessor():
    @staticmethod
    def process_start_point(image_lock, debug_lock, new_frame_event, FP_FPtoIR, img_shape, run_event, **kwargs):
        instance = _FrameProcessor(image_lock, debug_lock, new_frame_event, FP_FPtoIR, img_shape, run_event, **kwargs)
    
    def _print(self, string):
        self.debug_lock.acquire()
        try:
            print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        finally:
            self.debug_lock.release()
    
    def __init__(self, image_lock, debug_lock, new_frame_event, FP_FPtoIR, img_shape, run_event, single_stage=True, **kwargs):
        self.proc_name = current_process().name
        if  self.proc_name != 'FrameProcessor':
            raise RuntimeError("FrameProcessor must be instantiated on a seperate process, please use 'create_frame_processor' instead")
        
        self.skel = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # self._stage = self._single_stage if single_stage else self._two_stage
        
        self.new_frame_event = new_frame_event
        self.image_lock = image_lock
        self.debug_lock = debug_lock
        self.FP_FPtoIR = FP_FPtoIR
        self.run_event = run_event
        
        self.img_shape = img_shape
        self.aspect = img_shape[1]/img_shape[0]
        
        self.sm_image = SharedMemory(name=SM_IMAGE_NAME)
        self.frame = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.sm_image.buf)
        
        self.sm_time = SharedMemory(name=SM_TIME_NAME)
        self.frame_time = np.ndarray((), dtype=np.uint64, buffer=self.sm_time.buf)
        
        self.w = self.img_shape[1]
        self.h = self.img_shape[0]
        
        self.kps = np.zeros((33,3), dtype=np.float32)
        self.wkps = np.zeros((33,3), dtype=np.float32)
        self.vis = np.zeros(33, dtype=bool)
        self.prev_vis = np.zeros(33,dtype=bool)
        self.velocity = np.zeros((33,3), dtype=np.float32)
        
        self.null_sent = False
        self.prev_to_ir = FPtoIR_Packet()
        
        
        self.run_event.wait()
        
        self._process_loop()
        
    # def _OEF(self, update_rois, frame_time):
        # rois               = self._rois[self.labels]
        # filter_MC          = self.filter_MC[self.labels]
        # filter_dMC         = self.filter_dMC[self.labels]
        # filter_beta        = self.filter_beta[self.labels]
        # filter_prev_t      = self.filter_prev_t[self.labels]
        # filter_drois       = self.filter_drois[self.labels]
        
        # rois[filter_prev_t==0] = update_rois[filter_prev_t==0]
        # filter_prev_t[filter_prev_t==0] = frame_time-1
        # t_e = (frame_time - filter_prev_t)*1e-9
        # dx = (update_rois - rois) / t_e
        # if ENABLE_OEF:
            
            
            # r = 2 * np.pi * filter_dMC * t_e
            # a_d = r / (r + 1)
            # dx_hat = a_d * dx + (1 - a_d) * filter_drois
            
            # cutoff = filter_MC + filter_beta * np.abs(dx_hat)
            # r = 2 * np.pi * filter_MC * t_e
            # a = r / (r + 1)
            # x_hat = a * update_rois + (1 - a) * rois
            
            # self._rois[self.labels] = x_hat
            # self.filter_drois[self.labels] = dx_hat
            # self.filter_prev_t[self.labels] = frame_time
        # else:
            # self._rois[self.labels] = update_rois
            # self.filter_drois[self.labels] = dx
            # self.filter_prev_t[self.labels] = frame_time

    def _process_loop(self):
        count = 0
        frame_wait_time = 0
        lock_time = 0
        loop_time = 0
        exec_time = 0
        
        to_ir = FPtoIR_Packet()
        self.prev_to_ir = to_ir
        if self.FP_FPtoIR.poll():
            self.FP_FPtoIR.send(to_ir)
            self.FP_FPtoIR.recv()
            self.null_sent = True
        
        while self.run_event.is_set():
            t0 = time.perf_counter()
            self.new_frame_event.wait()
            frame_wait_time += (time.perf_counter()-t0)
            
            t = time.perf_counter()
            self.image_lock.acquire()
            lock_time += (time.perf_counter()-t)
            try:
                in_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                frame_time = self.frame_time.copy()
            finally:
                self.image_lock.release()
                self.new_frame_event.clear()
            
            t = time.perf_counter()
            in_frame.flags.writeable = False
            
            skel = self.skel.process(in_frame)
            
            if not skel or not skel.pose_landmarks:
                if not self.null_sent:
                    to_ir = FPtoIR_Packet()
                    self.prev_to_ir = to_ir
                    if self.FP_FPtoIR.poll():
                        self.FP_FPtoIR.send(to_ir)
                        if self.run_event.is_set(): self.FP_FPtoIR.recv()
                        self.null_sent = True
                continue
            
            for i, (kp, wkp) in enumerate(zip(skel.pose_landmarks.landmark,skel.pose_world_landmarks.landmark)):
                self.prev_vis[i] = self.vis[i]
                self.vis[i] = (kp.visibility>0.4)
                self.kps[i,0] = kp.x
                self.kps[i,1] = kp.y
                self.kps[i,2] = kp.z
                
                self.velocity[i,0] = self.wkps[i,0]-wkp.x
                self.velocity[i,1] = self.wkps[i,1]-wkp.y
                self.velocity[i,2] = self.wkps[i,2]-wkp.z
                self.wkps[i,0] = wkp.x
                self.wkps[i,1] = wkp.y
                self.wkps[i,2] = wkp.z
            
            viso = np.logical_and(self.vis, self.prev_vis)
            good_vis = viso[23] and viso[24]
            
            centered_u = self.wkps[CON_DUPLEX[0]] - self.wkps
            centered_v = self.wkps[CON_DUPLEX[1]] - self.wkps
            
            # centered_u = centered_u/np.linalg.norm(centered_u,axis=-1,keepdims=True)
            # centered_v = centered_v/np.linalg.norm(centered_v,axis=-1,keepdims=True)
            
            normals = np.cross(centered_u, centered_v)
            # normals[:,2] = np.abs(normals[:,2])
            normals = normals/np.linalg.norm(normals,axis=-1,keepdims=True)
            
            cnt = (self.wkps[viso] - np.mean(self.wkps[viso],axis=0,keepdims=True)).T
            cntplane = np.linalg.svd(cnt)[0][:,-1]
            cntplane = cntplane/np.linalg.norm(cntplane)
            cntplane[2] = np.abs(cntplane[2])
            
            to_ir = FPtoIR_Packet(kps=self.kps, vels=self.velocity, vis=viso, good_vis=good_vis, normals=normals, plane = cntplane)
            
            exec_time += (time.perf_counter()-t)
            
            
            if self.FP_FPtoIR.poll():
                self.FP_FPtoIR.send(to_ir)
                if self.run_event.is_set(): self.FP_FPtoIR.recv()
                self.null_sent = False
                
            self.prev_to_ir = to_ir
            
            
            # self.RGB_pipe.send(RGB_out)
            count +=1
            loop_time += (time.perf_counter()-t0)
        
        self._print(f"FP Send count: {count}")
        self._print(f"FP Execution time: {exec_time/count}")
        self._print(f"FP Frame wait: {frame_wait_time/count}")
        self._print(f"FP Lock wait: {lock_time/count}")
        self._print(f"FP Total time: {loop_time/count}")
        self.skel.close()
        self.sm_image.close()
    