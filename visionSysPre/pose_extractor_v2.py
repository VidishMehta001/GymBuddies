import cv2
import numpy as np
import math
import time
import mediapipe as mp
import matplotlib
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
matplotlib.use('TkAgg')
import random
import exercise_classifier_run

class PoseExtractor:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

        # This is the intrinsic matrix for the right depth camera.
        M_right = np.array([[856.140625, 0.000000, 633.547363], [0.000000, 856.490723, 401.511566], [0.000000, 0.000000, 1.000000]])
        M_right[1, 2] -= 40
        self.K_inv = np.linalg.inv(M_right)

        # Adjustment of depth for left/right most side of middle pixel ie 640px from center
        fullwidth_angle_depth_adjustment = math.cos(math.radians(73.5 / 2))
        angle_depth_pixel_adj = (1 - fullwidth_angle_depth_adjustment) / 640
        self.fullwidth_angle_depth_adjustment=fullwidth_angle_depth_adjustment
        self.angle_depth_pixel_adj=angle_depth_pixel_adj

        self.kps_norm_val = None
        self.rotation_matrix = None
        self.baseline_set = False
        self.depth_filter = False
        self.pose_queue = []
        self.pose_queue_size = 5
        self.fig,self.ax1,self.ax2 = None, None, None
        self.keypairs = [(0, 1), (0, 2), (2, 4), (0, 6), (1, 3), (3, 5), (1, 7), (6, 7), (6, 8), (7, 9), (8, 10),
                         (9, 11)]
        self.keypairs_alt = [(0, 2), (2, 4), (0, 6), (6, 8), (8, 10)]
        self.hold_kps = None

    def create_depth(self,dispartiy):
        # Convert disparity map to depth
        depth_map = 64279.5/dispartiy #75 * 857.06/dispartiy
        depth_map[depth_map==np.inf]=0
        depth_map[depth_map == 0] = np.nan
        return depth_map

    def create_pose(self,image):
        # Extracts pose estimation from image
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
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

    def extract_pose(self, depth_map, results):

        # Extract only relevent key-points of the user
        # [0:left_shoulder, 1:right_shoulder, 2:left_elbow, 3:right_elbow, 4:left_wrist, 5:right_wrist,
        # 6:left_hip, 7:right_hip, 8:left_knee, 9:right_knee, 10:left_ankle, 11:right_ankle]
        x = [results.pose_landmarks.landmark[i].x for i in range(len(results.pose_landmarks.landmark)) if
             i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]
        y = [results.pose_landmarks.landmark[i].y for i in range(len(results.pose_landmarks.landmark)) if
             i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]

        x = np.array(x) * 1280
        y = np.array(y) * 720

        z=[]
        for i in range(len(x)):
            # Catch instances where the coordinates are out of the image frame
            try:
                if not self.depth_filter:
                    current_z = depth_map[int(y[i]),int(x[i])]
                else:
                    current_z = np.nanmax(depth_map[int(y[i])-3:int(y[i])+3, int(x[i])-3:int(x[i])+3])
                if True: #Offset calculation based on keypoint distance from centre
                    dist_from_centre = 640 - ((y[i] - 360) ** 2 + (x[i] - 640) ** 2) ** 0.5
                    distance_adjustment_factor = 0.8012538126910607 + 0.0003105409176702177 * dist_from_centre
                    current_z = current_z * distance_adjustment_factor
                x[i], y[i], zi = np.matmul(self.K_inv, np.array([x[i], y[i], 1]) * current_z)
                z.append(current_z)
            except:
                z.append(0)
        z = np.array(z)
        x,y,z = self.translate_pose(x,y,z)
        y = -y
        kps = np.stack([x,y,z])
        kps = self.normalize(kps)
        return kps

    def normalize(self, kps):
        # Normalizes the data
        kps = self.check_nan(kps)
        if not self.kps_norm_val:
            self.kps_norm_val = (sum(kps[:,0]**2)**0.5)
        if self.kps_norm_val == 0:
            return kps*0
        return np.nan_to_num(kps/self.kps_norm_val)

    def translate_pose(self,x,y,z):
        x = x - x[6]
        y = y - y[6]
        z = z - z[6]
        return x,y,z

    def rotate_pose(self,kps):
        if type(self.rotation_matrix) == type(None):
            #an = -np.arctan(abs(kps[0,9] - kps[0,8]) / abs(kps[2,9] - kps[2,8]))
            if abs(kps[0,7])>0.9:
                an = 0.785398
            else:
                an = 0
            self.rotation_matrix = np.array([[np.cos(an), 0, np.sin(an)], [0, 1, 0], [-np.sin(an), 0, np.cos(an)]])
        kps = np.matmul(self.rotation_matrix, kps)
        return kps

    def draw_pose(self,image,results):
        # Draws Blazepose on image
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def median_filter(self, kps):
        self.pose_queue.append(kps)
        if len(self.pose_queue) > self.pose_queue_size:
            self.pose_queue.pop(0)
        filtered_kps = np.median(self.pose_queue, axis = 0)
        return filtered_kps

    def mirror_pose(self,kps,waist_val=0.4):
        for i in range(len(kps[0])):
            if i%2==0 or i==0:
                continue
            else:
                kps[0,i] =kps[0,i - 1]
                kps[1,i] =kps[1,i - 1]
                z_min = min([kps[2,i],kps[2,i-1]])
                kps[2,i-1]=z_min
                kps[2,i] = (-1) * z_min + waist_val
                #kps[2, i] = (-1) * kps[2,i-1] + waist_val
        return kps

    def check_nan(self,kps):
        if type(self.hold_kps) == type(None):
            self.hold_kps = kps
        for i in range(len(kps[0])):
            if np.isnan(kps[0, i]):
                kps[0, i] = self.hold_kps[0, i]
            if np.isnan(kps[1,i]):
                kps[1,i] = self.hold_kps[1,i]
            if np.isnan(kps[2,i]):
                kps[2,i] = self.hold_kps[2,i]
        self.hold_kps = kps
        return np.nan_to_num(kps)

    def image_keypoints(self, results):
        x = [results.pose_landmarks.landmark[i].x for i in range(len(results.pose_landmarks.landmark)) if i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]
        y = [results.pose_landmarks.landmark[i].y for i in range(len(results.pose_landmarks.landmark))if i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]
        x = np.array(x) * 1280
        y = np.array(y) * 720
        return x,y

    def run_pose_extract(self, results, depth_map, frame):
        success = True
        if frame > 100:
            kps = self.extract_pose(depth_map, results)
            if np.sum(kps)==0:
                success = False
            else:
                kps = self.rotate_pose(kps)
                kps = self.median_filter(kps)
                kps = self.mirror_pose(kps)
        else:
            success = False
            kps = [None,None,None]
        return success, kps[0], kps[1], kps[2]

    def plot_pose(self, x, y, z):
        if not self.fig:
            self.fig = plt.figure(figsize=(12, 5.5))
            self.ax1, self.ax2 = self.fig.add_subplot(121), self.fig.add_subplot(122)
        self.ax1.cla()
        self.ax2.cla()
        for pair in self.keypairs:
            x_pair = [x[pair[0]], x[pair[1]]]
            y_pair = [y[pair[0]], y[pair[1]]]
            z_pair = [z[pair[0]], z[pair[1]]]
            self.ax1.plot(x_pair, y_pair)
            self.ax2.plot(z_pair, y_pair)
        self.ax1.set_xlim([-2, 2])
        self.ax1.set_ylim([-2, 2])
        self.ax2.set_xlim([-2, 2])
        self.ax2.set_ylim([-2, 2])
        plt.pause(0.001)

    def plot_pose_3d(self,x,y,z, colu=(1, 0, 0)):
        if not self.fig:
            self.fig = plt.figure(figsize =(6,6))
            self.ax1 = plt.axes(projection='3d')
        self.ax1.cla()
        for pair in self.keypairs:
            x_pair = [x[pair[0]], x[pair[1]]]
            y_pair = [y[pair[0]], y[pair[1]]]
            z_pair = [z[pair[0]], z[pair[1]]]
            self.ax1.plot3D(x_pair, y_pair, z_pair, color=colu)
        self.ax1.set_xlim([-2, 2])
        self.ax1.set_ylim([-2, 2])
        self.ax1.set_zlim([-2, 2])

        plt.pause(0.001)

if __name__ == '__main__':

    # Video Selection
    location = './Video Saving/data/'
    exercise = 'squats_45'
    version = '8'
    new_loc = location + exercise + '/v' + version + '_' + exercise + '_'

    vp = PoseExtractor()
    vp.depth_filter = True
    frame_rate = 30
    prev = 0
    frame = 0
    fvs = FileVideoStream(new_loc + 'mono2.mp4').start()
    fvs1 = FileVideoStream(new_loc + 'mono1.mp4').start()
    fvs2 = FileVideoStream(new_loc + 'depth.mp4').start()
    saved_pose = []
    ec = exercise_classifier_run.ClassifyExercise(stride = 60)
    while fvs.more():
        # Setting to 30 FPS
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            #Reading the images
            img_mono = fvs.read()
            img_depth = fvs2.read()

            #Attempting to create the depth map
            try:
                depth_map =vp.create_depth(img_depth[:,:,0])
            except:
                break

            pose_success, pose_image, results = vp.create_pose(img_mono)

            id = cv2.applyColorMap(img_depth, cv2.COLORMAP_JET)
            if pose_success:
                success,x,y,z = vp.run_pose_extract(results, depth_map, frame)
                pose_image = vp.draw_pose(pose_image, results)
                id = vp.draw_pose(id, results)
                if success:
                    vp.plot_pose(x,y,z)
                    exercise_type = ec.predict(x,y,z)
                    print(exercise_type)
                    saved_pose.append([x, y, z])
            cv2.imshow('Depth', id)
            cv2.imshow('Pose', pose_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            frame += 1
        else:
            print('ITS TO FAST! VROOOOOOOOM')