import pose_extractor_v2
import Rules_engine  # import CheckRules as re
from imutils.video import FileVideoStream
from Key_Frame_Counter import KeyFrame_Counter_FeatureMap
from data_preprocessing import DataPreprocessing as dp
from hog_svm_classifier import HogSvmClassifier
from ExerciseClassification import ExerciseClassifier
import time
import cv2
import matplotlib.pyplot as plt
import exercise_classifier_run
import tensorflow as tf
import os

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
counts_total = []
v_counts_total = []
print('Starting tests...')
location = './Video Saving/data/'
for exercise in os.listdir(location):
    print('Running test for: ', exercise)
    vids = os.listdir(location+exercise)
    vid_list = [vids[i][:2] for i in range(len(vids)) if i % 4 == 0]
    for vid in vid_list:
        new_loc = location + exercise + '/' + vid + '_' + exercise + '_'

        vp = pose_extractor_v2.PoseExtractor()
        ec = exercise_classifier_run.ClassifyExercise(stride=10)
        re = Rules_engine.CheckRules()
        hsc = HogSvmClassifier()
        frame_rate = 300000
        prev = 0
        frame = 0
        saved_pose, Gradient_X, Gradient_Y, Dist, Feature_Map = [], [], [], [], []
        j = 0
        counter = 0
        ref, predict = False, False
        model = "CNN"
        pred = ""
        brokenRules = []
        exercises = ['No Exercise', 'squats', 'pushups']
        writeNotes = True
        timestart = 0
        timeend = 0
        decay = 0
        decaymsg = ''
        totaltime = 0
        pred_text = ""
        countVoid = False

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
                    # Set Success Manually - request to set it as true only after baseset
                    if success:
                        saved_pose.append([x, y, z])
                        # Set Reference
                        if ref == False:
                            refx = x
                            refy = y
                            refz = z
                            ref = True

                        # Set sampled frames
                        if pred_text == "squats":
                            sampled_frame, j = dp(n_samp=45).frame_sampling(j)
                        else:
                            sampled_frame, j = dp(n_samp=30).frame_sampling(j)

                        # counter_val, key_frame_detected, Feature_map = Key-frame/Counter Module > (int, int, array) - SIDE VIEW ONLY #
                        # Key frame 0 -- not a key frame ; 1 - maxima ; 2 - minima
                        counter, key_frame_detected, Feature_map = KeyFrame_Counter_FeatureMap(x=x, y=y, refx=refx,
                                                                                               refy=refy, Dist=Dist,
                                                                                               Gradient_X=Gradient_X,
                                                                                               Gradient_Y=Gradient_Y,
                                                                                               Feature_Map=Feature_Map,
                                                                                               Counter=counter,
                                                                                               sampled_frame=sampled_frame).key_frame_counter_fm()

                        exercise_type = ec.predict(x, y, z)
                        if not type(exercise_type) == type(None):
                            pred_text = exercises[exercise_type]
                            cv2.putText(pose_image, pred_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255),
                                        2)

                        # img, counter_neg, rules = Rules(keypoints, img)  > (img, bool, array)
                        if "squats" in pred_text:
                            pose_image, countVoid, brokenRules, decay, decaymsg, f_count = re.check_squat_rules(image_c,
                                                                                                                x, y, z,
                                                                                                                x_img,
                                                                                                                y_img,
                                                                                                                key_frame_detected,
                                                                                                                counter,
                                                                                                                brokenRules,
                                                                                                                writeNotes,
                                                                                                                decay,
                                                                                                                decaymsg)
                        if "pushup" in pred_text:
                            pose_image, countVoid, brokenRules, decay, decaymsg, f_count = re.check_pushup_rules(
                                image_c, x, y, z, x_img, y_img, key_frame_detected, counter, brokenRules, writeNotes,
                                decay, decaymsg)

                        # Write output to file
                        if countVoid == True and False:
                            filename_rules = './Output/' + str(frame) + ".jpg"
                            cv2.imwrite(filename_rules, pose_image)

                frame += 1
            else:
                print('ITS TO FAST! VROOOOOOOOM')

        counts_total.append(counter)
        v_counts_total.append(f_count)

print(counts_total)
print(v_counts_total)




