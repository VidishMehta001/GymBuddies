import os
import numpy as np
import cv2
from skimage import exposure

blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
IMG_WIDTH= 640
IMG_HEIGHT = 416
hog = cv2.HOGDescriptor((IMG_WIDTH, IMG_HEIGHT), blockSize, blockStride, cellSize, nbins)
dataset = []
labels = []
fileno=0

for file in os.listdir('./Video Saving/HOG_vid'):
    cap = cv2.VideoCapture('./Video Saving/HOG_vid/'+file)
    fileno += 1
    print(fileno)
    frame = 0
    print(file)

    if 'squats' in file:
        label = 1
    elif 'pushups' in file:
        label = 2
    elif 'other' in file:
        label = 0

    while cap.isOpened():
        success, img = cap.read()

        if np.shape(img) == ():
            print('empty')
            break
        else:
            if frame > 90 and frame < 660 and frame%10 == 0:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img_hog = hog.compute(img)
                #img_hog = img_hog/np.linalg.norm(img_hog)
                img_hog = exposure.rescale_intensity(img_hog, out_range=(0, 255)).astype("uint8")
                print(img_hog)
                #cv2.imshow('img_hog', img_hog)
                #print(frame)
                dataset.append(img_hog)
                labels.append(label)

        #cv2.imshow('Img', img)
        cv2.waitKey(1)
        #cap.release()
        #break

        frame += 1


outfile_data = './Video Saving/hog_data'
outfile1_label = './Video Saving/hog_label'
dataset_np = np.save(outfile_data, dataset)
labels_np = np.save(outfile1_label, labels)


