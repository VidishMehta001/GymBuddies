import pose_extractor_v2
from imutils.video import FileVideoStream
import time
import exercise_classifier_run
import os
import numpy as np
vp= pose_extractor_v2.PoseExtractor()
ec = exercise_classifier_run.ClassifyExercise(stride = 10)
frame_rate = 3000
prev = 0
location = './Video Saving/data/'

total_stack = []
total_true_stack = []

total_detailed_stack = []
total_detailed_true_stack = []

print('Starting tests...')
for exercise in os.listdir(location):
    print('Running test for: ', exercise)
    vids = os.listdir(location+exercise)
    vid_list = [vids[i][:2] for i in range(len(vids)) if i % 4 == 0]
    if 'squats' in exercise:
        t_id = 1
    elif 'pushups' in exercise:
        t_id = 2
    else:
        t_id = 0
    for vid in vid_list:
        frame = 0
        new_loc = location + exercise + '/' + vid + '_' + exercise + '_'
        fvs = FileVideoStream(new_loc + 'mono2.mp4').start()
        fvs2 = FileVideoStream(new_loc + 'depth.mp4').start()

        current_exercise_stack = []
        while fvs.more():
            time_elapsed = time.time() - prev
            if time_elapsed > 1. / frame_rate:
                prev = time.time()
                image_c = fvs.read()
                image_d = fvs2.read()
                try:
                    depth_map = vp.create_depth(image_d[:, :, 0])
                except:
                    break
                pose_success, pose_image, results = vp.create_pose(image_c)

                if pose_success:
                    success, x, y, z = vp.run_pose_extract(results, depth_map, frame)
                    x_img, y_img = vp.image_keypoints(results)  # image keypoints for Chun How
                    if success:
                        exercise_type = ec.predict(x, y, z)
                        if not exercise_type  == None:
                            current_exercise_stack.append(exercise_type)
                            if frame > 180 and frame < 600:
                                total_detailed_stack.append(exercise_type)
                                total_detailed_true_stack.append(t_id)
                            elif 'other' in exercise:
                                total_detailed_stack.append(exercise_type)
                                total_detailed_true_stack.append(t_id)

            #cv2.imshow('Pose', pose_image)
            #if cv2.waitKey(5) & 0xFF == 27:
            #    break
            frame += 1
        vals, counts = np.unique(current_exercise_stack, return_counts=True)
        index = np.argmax(counts)
        total_stack.append(vals[index])
        total_true_stack.append(t_id)


overall_cm = np.array([[0,0,0],[0,0,0],[0,0,0]])
for i in range(len(total_stack)):
    overall_cm[total_true_stack[i],total_stack[i]]+=1
print('Rows = Actual, Columns = Predicted')
print(overall_cm)
#total_stack > [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#total_true_stack > [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#[[ 2,  0,  0],
#[ 0, 20,  0],
#[ 0,  0, 20]]

overall_cm_detailed = np.array([[0,0,0],[0,0,0],[0,0,0]])
for i in range(len(total_detailed_stack)):
    overall_cm_detailed[total_detailed_true_stack[i],total_detailed_stack[i]]+=1

print('Rows = Actual, Columns = Predicted')
print(overall_cm_detailed)

vid_list_names = ['./Video Saving/data/squats/v1_squats_',
                  './Video Saving/data/squats/v3_squats_',
                  './Video Saving/data/squats_45/v1_squats_45_',
                  './Video Saving/data/squats_45/v3_squats_45_',
                  './Video Saving/data/pushups/pushups',
                  './Video Saving/data/pushups/v3_pushups_',
                  './Video Saving/data/pushups/v1_pushups_45_',
                  './Video Saving/data/pushups/v3_pushups_45_',
                  './Video Saving/data/other/v1_other',
                  './Video Saving/data/other/v2_other',
                  ]

print('Starting Timed Test...')
time_start = time.time()
total_frames  =  0
for new_loc in vid_list_names:
    frame = 0
    fvs = FileVideoStream(new_loc + 'mono2.mp4').start()
    fvs2 = FileVideoStream(new_loc + 'depth.mp4').start()
    while fvs.more():
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            image_c = fvs.read()
            image_d = fvs2.read()
            try:
                depth_map = vp.create_depth(image_d[:, :, 0])
            except:
                break
            pose_success, pose_image, results = vp.create_pose(image_c)

            if pose_success:
                success, x, y, z = vp.run_pose_extract(results, depth_map, frame)
                x_img, y_img = vp.image_keypoints(results)  # image keypoints for Chun How
                if success:
                    exercise_type = ec.predict(x, y, z)
        frame += 1
    total_frames += frame
total_time = time.time() - time_start
print('Avg Time per Video =  ', total_time/10)
print('Avg Time per Frame =  ', total_time/total_frames)
print('Avg GPS =  ', 1/(total_time/total_frames))
#Avg. Time per Video = 16.45306510925293
#Avg Time  per Frame = 0.04175904850064195
#Avg FPS = 23.946905782219325