import cv2
import numpy as np
import math
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
import random

class PoseExtractor:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # This is the intrinsic matrix for the right depth camera.
        M_right = np.array([[856.140625, 0.000000, 633.547363], [0.000000, 856.490723, 401.511566], [0.000000, 0.000000, 1.000000]])
        M_right[1, 2] -= 40
        self.K_inv = np.linalg.inv(M_right)

        # Adjustment of depth for left/right most side of middle pixel ie 640px from center
        fullwidth_angle_depth_adjustment = math.cos(math.radians(73.5 / 2))
        angle_depth_pixel_adj = (1 - fullwidth_angle_depth_adjustment) / 640
        self.fullwidth_angle_depth_adjustment=fullwidth_angle_depth_adjustment
        self.angle_depth_pixel_adj=angle_depth_pixel_adj


        self.xlist=np.array([])
        self.ylist=np.array([])
        self.zlist=np.array([])
        self.x_norm_min = 0
        self.x_norm_ptp = 0
        self.pose_adj = None
        self.set_baseline = False
        self.augment_matrix = []
        self. augment = False

    def create_depth(self,dispartiy):
        # Convert disparity map to depth
        depth_map = 75 * 856.140625/dispartiy/255*95
        depth_map[depth_map==np.inf]=0
        depth_map[depth_map == 0] = np.nan
        return depth_map

    def create_mp_pose(self,image):
        # Extracts pose estimation from image
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks:
            success=1
        else:
            success=0
        return success, image, results

    def draw_pose(self,image,results):
        # Draws Blazepose on image
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def extract_pose(self, depth_map, results, depth_lock_val = None, fs=9, fk=11):
        # Extracts the pose information and gets the depth from the x,y coords. Transforms the x,y coords to camera
        # reference system and recalculates the depth.

        x = [results.pose_landmarks.landmark[i].x for i in range(len(results.pose_landmarks.landmark))]
        y = [results.pose_landmarks.landmark[i].y for i in range(len(results.pose_landmarks.landmark))]
        x=np.array(x)*1280
        y=np.array(y)*720
        mfactor=[]
        z=[]
        z_r=[]

        for i in range(len(x)):
            if x[i] >=1280 or x[i]<0 or y[i]>=720 or y[i]<0:
                z_r.append(0)
                z.append(0)
                mfactor.append(0)
                continue

            dist = 640 - ((y[i] - 360) ** 2 + (x[i] - 640) ** 2) ** 0.5
            mfactor.append(self.fullwidth_angle_depth_adjustment+self.angle_depth_pixel_adj*dist)
            z_r.append(np.nanmin(depth_map[int(y[i])-fs:int(y[i])+fs, int(x[i])-fs:int(x[i])+fs]))
        z_r[25]=np.nanmin(depth_map[int(y[25])-fk:int(y[25])+fk, int(x[25])-fk:int(x[25])+fk])

        for i in range(len(x)):
            #z_1 = z_r[i] * mfactor[i]
            #x[i], y[i], zi = np.matmul(self.K_inv, np.array([x[i], y[i], 1]) * z_1)
            #z.append(zi)
            x[i],y[i],zi=np.matmul(self.K_inv,np.array([x[i],y[i],1])*z_r[i])
            z.append(zi * mfactor[i])
        z=np.array(z)

        # Checks if the pose needs to be rotated, e.g. person is standing at an angle
        x,y,z = self.rotate_pose(x,y,z)

        #adjusts the pose with rules and creates some fixed values.
        z = self.check_and_adjust_pose(z, depth_lock_val)
        self.append_list(x, y, z)

        # Required to smoothen the output
        x = np.average(self.xlist, axis=0)
        y = np.average(self.ylist, axis=0)
        z = np.average(self.zlist, axis=0)

        z_r = np.array(z_r)

        return x,y,z,z_r

    def rotate_pose(self, x, y, z):
        x = x - x[23]
        y = y - y[23]
        z = z - z[23]
        if type(self.pose_adj) == type(None):
            if abs(x[23]-x[24])<120:
                self.pose_adj = [[1,0,0],[0,1,0],[0,0,1]]
            else:
                an = -np.arctan((x[24]-x[23])/(z[24]-z[23]))
                self.pose_adj = np.array([[np.cos(an), 0, np.sin(an)],[0,1,0],[-np.sin(an), 0, np.cos(an)]])
        ret = np.matmul(self.pose_adj, np.array([x, y, z]))
        x = ret[0]
        y = ret[1]
        z = ret[2]
        return x,y,z

    def check_and_adjust_pose(self, z, depth_lock_val):
        if self.zlist.size > 0:
            avg1_check = []
            avg_check = []
            previous_z = self.zlist[-1]
            z_avg = np.average(np.vstack((self.zlist, z)), axis=0)
            z_avg1 = np.average(self.zlist, axis=0)

            for limb in [13, 15, 25]:
                if np.abs(z[limb] - previous_z[limb]) > 1500:
                    z[limb] = z_avg1[limb]
                    avg1_check.append(limb)

                if np.abs(z[limb] - previous_z[limb]) > 300:
                    z[limb] = z_avg[limb]
                    avg_check.append(limb)

        # Checks if the shoulders are wider than the hips
        if z[11] > z[23] and self.zlist.size > 0:
            if z_avg[11] < z[23]:
                z[11] = z_avg[11]
            elif z_avg1[11] < z[23]:
                z[11] = z_avg1[11]
            else:
                z[11] = z[23]
        # Checks if shoulders are too wide
        if z[11] + 250 < z[23]:
            z[11] = z[23] - 250

        # Checks if ankles are too wide
        if z[11] + 150 < z[27]:
            z[27] = z[11]

        # Checks if other parts of the body is past the  mid point
        for limb in [13, 15, 25]:
            if type(depth_lock_val) != type(None) and z[limb] - 150 > depth_lock_val[23]:
                if self.zlist.size > 0 and z_avg[limb] - 150 < depth_lock_val[23]:
                    z[limb] = z_avg[limb]
                elif self.zlist.size > 0 and z_avg1[limb] - 150 < depth_lock_val[23]:
                    z[limb] = z_avg1[limb]
                else:
                    z[limb] = depth_lock_val[23] - 100

        # Checks if knees are too wide
        if type(depth_lock_val) != type(None) and z[25] + 500 < depth_lock_val[23]:
            if self.zlist.size > 0 and z_avg[25] + 500 > depth_lock_val[23]:
                z[25] = z_avg[25]
            elif self.zlist.size > 0 and z_avg1[25] - 500 > depth_lock_val[23]:
                z[25] = z_avg1[25]
            else:
                z[25] = depth_lock_val[23] - 500

        # Locks the shoulder, hip and ankle depth. Which should not move. THIS IS A ASSUMPTION.
        if type(depth_lock_val) != type(None):
            z[11] = depth_lock_val[11]
            z[23] = depth_lock_val[23]
            z[27] = depth_lock_val[27]

        return z

    def extract_pose_2d(self, results):
        # Simpler extract for 2d poses
        x = [results.pose_landmarks.landmark[i].x for i in range(len(results.pose_landmarks.landmark))]
        y = [results.pose_landmarks.landmark[i].y for i in range(len(results.pose_landmarks.landmark))]
        x=np.array(x)*1280
        y=np.array(y)*720
        self.append_list_2d(x, y)
        x = np.average(self.xlist, axis=0)
        y = np.average(self.ylist, axis=0)
        return x,y

    def append_list(self, x, y, z, max_len=5):
        # Updates the historical x,y,z for filtering for max_len previous frames.
        if self.xlist.size == 0:
            self.xlist = np.array(x)
            self.ylist = np.array(y)
            self.zlist = np.array(z)

        self.xlist = np.vstack((self.xlist, x))
        self.ylist = np.vstack((self.ylist, y))
        self.zlist = np.vstack((self.zlist, z))

        if len(self.xlist) > max_len:
            self.xlist = self.xlist[1:]
            self.ylist = self.ylist[1:]
            self.zlist = self.zlist[1:]

    def append_list_2d(self,x,y,max_len=5):
        # Updates the historical x,y for filtering for max_len previous frames.
        if self.xlist.size ==0:
            self.xlist = np.array(x)
            self.ylist = np.array(y)

        self.xlist=np.vstack((self.xlist,x))
        self.ylist=np.vstack((self.ylist,y))

        if len(self.xlist)>max_len:
            self.xlist=self.xlist[1:]
            self.ylist = self.ylist[1:]

    def zero_origin_mirror(self,x,y,z,waist_val=300):
    # Sets the right hip to the origin and mirrors
        for i in range(len(x)):
            if not i%2==0 or i==0:
                continue
            else:
                x[i] =x[i - 1]
                y[i] =y[i - 1]
                z[i] = (-1) * z[i - 1] + waist_val
        return x,y,z

    def normalize(self, x):
        # Normalizes the data and reduces it to only include shoulder, elbow, wrist, hip ,knee and ankle
        x = np.asarray(x)
        x = x[:,(11,12,13,14,15,16,23,24,25,26,27,28)]
        if not self.x_norm_min:
            self.x_norm_min = x.min()
            self.x_norm_ptp = np.ptp(x)

        return ((x - self.x_norm_min) / self.x_norm_ptp)

    def checkVertHor(self, results):
        # Checks if person is horizontal or vertical which is needed to select how to rotate the detected keypoints.
        self.vertical = True
        x = [results.pose_landmarks.landmark[i].x for i in range(len(results.pose_landmarks.landmark))]
        y = [results.pose_landmarks.landmark[i].y for i in range(len(results.pose_landmarks.landmark))]
        if np.std(x) > np.std(y):
            self.vertical = False

    def depth_lock(self, results, depth_map):
        x1,y1,z1,z_r = self.extract_pose(depth_map, results)
        self.depth_lock_v = z1
        return x1,y1,z1

    def run_pose_extract(self, results, depth_map, frame):
        success = True
        if not self.set_baseline and frame > 100:
            #self.checkVertHor(results)
            x,y,z = self.depth_lock(results, depth_map)
            self.depth_val_z = z
            self.set_baseline = True
            # Zero origin and Create Rotation Function
            success = False
            x = None
            y = None
            z = None
        elif self.set_baseline and frame > 100:
            try:
                x1, y1, z1, z_r = self.extract_pose(depth_map, results, self.depth_val_z)
                x1, y1, z1 = self.zero_origin_mirror(x1, y1, z1)
                [x, y, z] = self.normalize([x1, y1, z1])
            except:
                success = False
                x = None
                y = None
                z = None
        else:
            success = False
            x= None
            y= None
            z= None
        return success,x,y,z

    def image_keypoints(self, results):
        x = [results.pose_landmarks.landmark[i].x for i in range(len(results.pose_landmarks.landmark)) if i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]
        y = [results.pose_landmarks.landmark[i].y for i in range(len(results.pose_landmarks.landmark))if i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]
        x = np.array(x) * 1280
        y = np.array(y) * 720

        return x,y

if __name__ == '__main__':

    # Video Selection
    location = './Video Saving/Videos/'
    exercise = 'squats'
    version = '3'
    new_loc = location + exercise + '/v' + version + '_' + exercise + '_'

    saved_pose = []

    vp = PoseExtractor()
    frame_rate = 30
    prev = 0
    frame = 0
    fvs = FileVideoStream(new_loc + 'mono2.mp4').start()
    fvs2 = FileVideoStream(new_loc + 'depth.mp4').start()
    while fvs.more():

        # Setting to 30 FPS
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            image_c=fvs.read()
            image_d = fvs2.read()
            try:
                depth_map =vp.create_depth(image_d[:,:,0])
            except:
                break
            pose_success, pose_image, results = vp.create_mp_pose(image_c)

            if pose_success:
                success,x,y,z = vp.run_pose_extract(results, depth_map, frame)
                x_img, y_img = vp.image_keypoints(results)

                if success:
                    saved_pose.append([x,y,z])

            id= cv2.applyColorMap(image_d, cv2.COLORMAP_JET)
            cv2.imshow('Depth', id)
            cv2.imshow('Pose', pose_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            frame += 1
        else:
            print('ITS TO FAST! VROOOOOOOOM')