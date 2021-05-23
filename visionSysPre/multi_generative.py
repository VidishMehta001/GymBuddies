import pose_extractor_v2
from imutils.video import FileVideoStream
from Key_Frame_Counter import KeyFrame_Counter_FeatureMap
from data_preprocessing import DataPreprocessing as dp
import time
import cv2
import os
import numpy as np
import random

vp= pose_extractor_v2.PoseExtractor()
frame_rate = 3000
prev = 0
frame = 0
location = './Video Saving/data/'
exercises = ['squats','squats_45','pushups','pushups_45','other']
j = 0
Counter = 0

for exercise in exercises:
    vids = os.listdir('./Video Saving/data/'+exercise)
    vid_list = [vids[i][:2] for i in range(len(vids)) if i%4==0]

    for vid in vid_list:
        new_loc = location + exercise + '/' + vid + '_' + exercise + '_'
        cap = cv2.VideoCapture(new_loc + 'mono2.mp4')
        cap2 = cv2.VideoCapture(new_loc + 'depth.mp4')
        saved_pose = []
        npy_name = vid + '_' + exercise+'.npy'
        print('Creating npy file: ',npy_name)
        if npy_name in os.listdir('./augmented_arrays/'):
            print(npy_name,' already created. Skipping video...')
            continue
        while cap.isOpened():
            time_elapsed = time.time() - prev
            if time_elapsed > 1. / frame_rate:
                prev = time.time()
                r, image_c=cap.read()
                r2, image_d = cap2.read()
                try:
                    depth_map =vp.create_depth(image_d[:,:,0])
                except:
                    break
                pose_success, pose_image, results = vp.create_pose(image_c)

                if pose_success:
                    success, x, y, z = vp.run_pose_extract(results, depth_map, frame)
                    #x_img, y_img = vp.image_keypoints(results)  # image keypoints for Chun How
                    # Set Success Manually - request to set it as true only after baseset
                    if success:
                        saved_pose.append([x,y,z])


                #cv2.imshow('Pose', pose_image)
                #if cv2.waitKey(5) & 0xFF == 27:
                #    break
                frame += 1
            else:
                print('ITS TO FAST! VROOOOOOOOM')

        consolidated = []
        saved_pose = np.array(saved_pose)
        consolidated.append(saved_pose)
        for i in range(4):
            rand_list = []
            for j in range(3):
                rand_sml_list = []
                for k in range(12):
                    rand_sml_list.append(random.randint(9700,10300)/10000)
                rand_list.append(rand_sml_list)
            rand_list = np.array(rand_list)

            new_pose = saved_pose * rand_list
            consolidated.append(new_pose)

        consolidated = np.array(consolidated)
        save_loc = './augmented_arrays/' + vid + '_' + exercise
        np.save(save_loc,consolidated)