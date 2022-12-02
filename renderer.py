########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from multiprocessing import current_process, Event, Process
from multiprocessing.shared_memory import SharedMemory
import time
import ctypes

import psutil
import numpy as np
import mediapipe as mp
import cv2
from scipy.ndimage import gaussian_filter

from camera import SM_IMAGE_NAME
from packets import FPtoIR_Packet
from util import gkern

WIN_NAME = "Skeleton Demo"
V_FIELD_MIN =50
GAUSS_RADIUS = 3

def create_image_renderer(image_lock, debug_lock, new_frame_event, FP_in, img_shape, **kwargs):
    
    if "ImageRenderer" in [p.name for p in psutil.process_iter()]:
        raise RuntimeError("ImageRenderer instance already running")
    
    run_event = Event()
    close_event = Event()
    process = Process(target=_ImageRenderer.process_start_point, args=[image_lock, debug_lock, new_frame_event, FP_in, img_shape, run_event, close_event],kwargs=kwargs, name="ImageRenderer", daemon=True)
    
    process.start()
    
    return process, run_event, close_event

class _ImageRenderer():
    @staticmethod
    def process_start_point(image_lock, debug_lock, new_frame_event, FP_in, img_shape, run_event, close_event, **kwargs):
        instance = _ImageRenderer(image_lock, debug_lock, new_frame_event, FP_in, img_shape, run_event, close_event, **kwargs)
    
    def _print(self, string):
        self.debug_lock.acquire()
        try:
            print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        finally:
            self.debug_lock.release()
    
    def __init__(self, image_lock, debug_lock, new_frame_event, FP_in, img_shape, run_event, close_event, **kwargs):
        self.proc_name = current_process().name
        if  self.proc_name != 'ImageRenderer':
            raise RuntimeError("ImageRenderer must be instantiated on a seperate process, please use 'create_image_renderer' instead")
        
        self.new_frame_event = new_frame_event
        self.image_lock = image_lock
        self.debug_lock = debug_lock
        self.FP_in = FP_in
        self.run_event = run_event
        self.close_event = close_event
        
        # self.colours = []
        # for c in range(32):
            # self.colours.append(np.squeeze(cv2.cvtColor(np.array([[[128, np.random.randint(0,256),np.random.randint(0,256)]]],dtype=np.uint8), cv2.COLOR_LAB2BGR)).tolist())
        
        self.img_shape = img_shape
        self.sm_image = SharedMemory(name=SM_IMAGE_NAME)
        self.frame = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.sm_image.buf)
        self.in_w = self.img_shape[1]
        self.in_h = self.img_shape[0]
        
        cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        user32 = ctypes.windll.user32
        cv2.resizeWindow(WIN_NAME, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        cv2.moveWindow(WIN_NAME,0,0)
        _, _, self.win_w, self.win_h = cv2.getWindowImageRect(WIN_NAME)
        self.out_image = np.zeros((self.win_h,self.win_w,3),dtype=np.uint8)
        
        in_d_w = self.in_w*2
        full_aspect = in_d_w/self.in_h
        
        if full_aspect>=1:
            out_h = self.win_h
            out_w = int(out_h * full_aspect)
            y_border = 0
            x_border = (self.win_w - out_w)//2
        else:
            out_w = self.win_w
            out_h = int(out_w / full_aspect)
            y_border = (self.win_h - out_h)//2
            x_border = 0
        
        self.resize = (out_w//2, out_h)
        
        self.skel_slice = np.s_[y_border:y_border+out_h,x_border:x_border+out_w//2]
        self.map_slice = np.s_[y_border:y_border+out_h,x_border+out_w//2:x_border+out_w]
        
        half_aspect = self.resize[0]/self.resize[1]
        if half_aspect>=1:
            vec_shape = (V_FIELD_MIN+GAUSS_RADIUS*2, int(V_FIELD_MIN*half_aspect)+4, GAUSS_RADIUS*2)
        else:
            vec_shape = (int(V_FIELD_MIN/half_aspect)+GAUSS_RADIUS*2,V_FIELD_MIN+GAUSS_RADIUS*2, 3)
            
        self.v_shape = np.array(vec_shape[:2])
        self.v_inner_shape = self.v_shape-GAUSS_RADIUS*2
        self.v_field = np.zeros(vec_shape)
        
        self.map_ = np.full((out_h, out_w//2, 3), [127,127,255], dtype=np.uint8)
        
        gw = (GAUSS_RADIUS*2)+1
        A = np.zeros((gw,gw))
        A[2,2]=gw*2
        self.yn = -gaussian_filter(A, 1,order=[1,0])
        self.xn = -gaussian_filter(A, 1,order=[0,1])
        
        # self.xn = np.array([
                            # [-0.2,0,0.2],
                            # [-1,0,1],
                            # [-0.2,0,0.2]
                        # ])
        # self.yn = np.array([
                            # [-0.2,-1,-0.2],
                            # [0,0,0],
                            # [0.2,1,0.2]
                        # ])
                        
        # x = np.arange(0,out_w//8)
        # y = np.arange(0,out_h//4)

        # self.x_grid, self.y_grid = np.meshgrid(x, y)
        
        # self.gl = np.min(self.resize) // 8
        # if self.gl%2==0:
            # self.gl = self.gl+1
            
        # self.gauss = gkern(self.gl, np.sqrt(self.gl)*3)*10
        
        # self.left = np.diff(self.gauss,axis=1)
        # self.down = np.diff(self.gauss,axis=0)
        
        self.prev_packet = FPtoIR_Packet()
        
        
        self.map_placeholder = self.map_.copy()
        
        tsz = np.min(self.resize)//200
        (tw, th), bs = cv2.getTextSize("Please move", cv2.FONT_HERSHEY_SIMPLEX,tsz, 8)
        plx = (self.map_.shape[1]-tw)//2
        ply = (self.map_.shape[0])//2 - bs
        self.map_placeholder = cv2.putText(self.map_placeholder, "Please move", (plx,ply),  cv2.FONT_HERSHEY_SIMPLEX,tsz, (255,255,0), 8)
        self.map_placeholder.flags.writeable = False
        (tw, th), bs = cv2.getTextSize("further back", cv2.FONT_HERSHEY_SIMPLEX, tsz, 8)
        plx = (self.map_.shape[1]-tw)//2
        ply = (self.map_.shape[0])//2 + th + bs
        self.map_placeholder = cv2.putText(self.map_placeholder, "further back", (plx,ply),  cv2.FONT_HERSHEY_SIMPLEX, tsz, (255,255,0), 8)
        self.map_placeholder.flags.writeable = False
        
        self.con_was_visible = np.zeros(len(mp.solutions.pose.POSE_CONNECTIONS),dtype=int)
        
        self.first_poll = True
        
        self.run_event.wait()
        self._process_loop()
        
    def _draw_packet_skel(self, packet, draw_img):
        if packet.kps is None:
            return draw_img
            
        drawx = (packet.kps[:,0] * self.resize[0]).astype(int)
        drawy = (packet.kps[:,1] * self.resize[1]).astype(int)
        
        for i,connection in enumerate(mp.solutions.pose.POSE_CONNECTIONS):
            if not packet.vis[connection[0]] or not packet.vis[connection[1]]:
                if self.con_was_visible[i]>0:
                    v = int((self.con_was_visible[i]/30)*255)
                    cv2.line(draw_img, (drawx[connection[0]], drawy[connection[0]]), (drawx[connection[1]], drawy[connection[1]]), (v,v,v), 1)
                    self.con_was_visible[i] -= 1
                continue
            cv2.line(draw_img, (drawx[connection[0]], drawy[connection[0]]), (drawx[connection[1]], drawy[connection[1]]), (255,255,255), 2)
            self.con_was_visible[i] = 30
        return draw_img
        
        
    def _update_map(self, packet):
        if packet.kps is None:
            self.map_ = self.map_placeholder
            return
        
        self.v_field*=0.5
        # self.v_field += packet.plane*0.05
        xp = (packet.kps[:,0] * self.v_inner_shape[1]).astype(int)
        yp = (packet.kps[:,1] * self.v_inner_shape[0]).astype(int)
        viso = packet.vis & (xp>=0) & (yp>=0) & (yp<self.v_inner_shape[0]-1) & (xp<self.v_inner_shape[1]-1)
        vels = packet.vels[viso]
        xp = xp[viso]+GAUSS_RADIUS
        yp = yp[viso]+GAUSS_RADIUS
        norms = packet.normals[viso]
        for x,y,n,v in zip(xp,yp,norms,vels):
            self.v_field[y-GAUSS_RADIUS:y+GAUSS_RADIUS+1,x-GAUSS_RADIUS:x+GAUSS_RADIUS+1,0] += (n[0]*self.xn+n[1]*self.yn)*0.3
            self.v_field[y-GAUSS_RADIUS:y+GAUSS_RADIUS+1,x-GAUSS_RADIUS:x+GAUSS_RADIUS+1,1] += (v[0]*self.xn+v[1]*self.yn)*4
        self.v_field[yp,xp,0] += norms[:,2]*0.3#norms*0.15
        self.v_field[yp,xp,1] += vels[:,2]*4
        self.map_ = cv2.resize((np.clip(self.v_field[GAUSS_RADIUS:-GAUSS_RADIUS,GAUSS_RADIUS:-GAUSS_RADIUS,:]+1,0,2)*127).astype(np.uint8),self.resize,interpolation=cv2.INTER_CUBIC)
        # mph, mpw = self.map_.shape[:2]
        # drawx = (packet.kps[:,0] * mpw//4).astype(int) 
        # drawy = (packet.kps[:,1] * mph//4).astype(int)
        # drawx = drawx[packet.vis]
        # drawy = drawy[packet.vis]
        # # v = packet.vels.copy()
        # # q = np.linalg.norm(q,axis=-1)
        # # v
        
        # print(np.max(np.abs(packet.vels[packet.vis])))
        
        # rbf = RBFInterpolator(np.stack([drawx, drawy],-1), packet.vels[packet.vis], kernel='cubic')
        # map_sm = rbf( np.stack([self.x_grid.ravel(), self.y_grid.ravel()],-1)).reshape((mph//4,mpw//4,3))
        # 

        #zscale = ((packet.kps[:,2]*mpw)/np.sqrt(np.square(drawx[23]-drawx[24])+np.square(drawy[23]-drawy[24])))
        # for x,y,v in zip(drawx,drawy,packet.vels):
            # isx = x-self.gl//2
            # isy = y-self.gl//2
            # iex = isx+self.gl
            # iey = isy+self.gl
            
            # if isx<0:
                # gsx = -isx
                # isx = 0
            # elif isx>=mpw:
                # continue
            # else:
                # gsx = 0
                
            # if isy<0:
                # gsy = -isy
                # isy = 0
            # elif isy>=mph:
                # continue
            # else:
                # gsy = 0
                
            # if iex>mpw:
                # gex = mpw-iex
                # iex = mpw
            # elif iex<=0:
                # continue
            # else:
                # gex = self.gl
                
            # if iey>mph:
                # gey = mph-iey
                # iey = mph
            # elif iey<=0:
                # continue
            # else:
                # gey = self.gl
            
            # # print(f"{isy}:{iey},{isx}:{iex}")
            # # print(f"{gsy}:{gey},{gsx}:{gex}")
            # self.tmp_map[isy:iey,isx:iex,0] += self.gauss[gsy:gey,gsx:gex] * v[2]
            # self.tmp_map[isy:iey,isx:iex,1] += self.gauss[gsy:gey,gsx:gex] * v[1]
            # self.tmp_map[isy:iey,isx:iex,2] += self.gauss[gsy:gey,gsx:gex] * v[0] * 2
        
        # self.map_ = cv2.resize((np.clip(map_sm * 2, -127, 127)+127).astype(np.uint8), self.resize, interpolation=cv2.INTER_CUBIC)
        
        
    def _process_loop(self):
        count = 0
        skip_count = 0
        recieve_count = 0
        
        frame_wait_time = 0
        lock_time = 0
        loop_time = 0
        
        self.FP_in.send(None)
        
        while self.run_event.is_set():
            t0 = time.perf_counter()
            self.new_frame_event.wait()
            frame_wait_time += (time.perf_counter()-t0)
            
            t = time.perf_counter()
            self.image_lock.acquire()
            lock_time += (time.perf_counter()-t)
            try:
                skel_img = cv2.resize(self.frame, self.resize, interpolation=cv2.INTER_CUBIC)
            finally:
                self.image_lock.release()
                self.new_frame_event.clear()
            
            count +=1
            if self.first_poll or self.FP_in.poll():
                packet = self.FP_in.recv()
                self.FP_in.send(None)
                if self.first_poll:
                    self.first_poll = False#packet.kps is not None
                    recieve_count = 0
                    continue
            
                recieve_count +=1
            
                skel_img = self._draw_packet_skel(packet, skel_img)
                if packet.good_vis:
                    self._update_map(packet)
                else:
                    # self.map_[:] = (127,127,127)
                    self.map_ = self.map_placeholder
                    
                self.prev_packet = packet
            else:
                skel_img = self._draw_packet_skel(self.prev_packet, skel_img)
                self.map_ = self.map_placeholder
                
            self.out_image[self.skel_slice] = skel_img
            self.out_image[self.map_slice] = self.map_
            cv2.imshow(WIN_NAME, self.out_image)
            # self.temp[:] = packet.temp
            # cv2.imshow("temp",self.temp)
            k = cv2.waitKey(1)
            if k==ord("q") or k==27:
                self.close_event.set()
            
            
            loop_time += (time.perf_counter()-t0)
        
        cv2.destroyAllWindows()
        self._print(f"IR Frame count: {count}")
        self._print(f"IR Recieve count: {recieve_count}")
        self._print(f"IR Frame wait: {frame_wait_time/count}")
        self._print(f"IR Lock wait: {lock_time/count}")
        self._print(f"IR Total time: {loop_time/count}")
        self.sm_image.close()
    