import cv2
import depthai as dai
import numpy as np
import os
import time
#time.sleep(3)
location='./data/'

exercise='squats'
#exercise='pushups'
#exercise='other'

version_array = []
file_array = os.listdir(location+squats)
for i in range(len(file_array)):
    version_array.append(int(file_array[i][1]))
version=str(max(version_array)+1)

new_loc=location+exercise+'/v'+version+'_'+exercise+'_'
median = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color and mono cameras
colorCam = pipeline.createColorCamera()
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

monoCam = pipeline.createMonoCamera()
monoCam.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoCam2 = pipeline.createMonoCamera()
monoCam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
# Note: the rectified streams are horizontally mirrored by default
depth.setOutputRectified(True)
#depth.setExtendedDisparity(True)
#depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
depth.setMedianFilter(median)
depth.setRectifyEdgeFillColor(0) # Black, to better see the cutout
monoCam.out.link(depth.left)
monoCam2.out.link(depth.right)

# Create encoders, one for each camera, consuming the frames and encoding them using H.264 / H.265 encoding
ve1 = pipeline.createVideoEncoder()
ve1.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
monoCam.out.link(ve1.input)

ve2 = pipeline.createVideoEncoder()
ve2.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
colorCam.video.link(ve2.input)

ve3 = pipeline.createVideoEncoder()
ve3.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
monoCam2.out.link(ve3.input)

ve4 = pipeline.createVideoEncoder()
ve4.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
depth.disparity.link(ve4.input)

# Create outputs
ve1Out = pipeline.createXLinkOut()
ve1Out.setStreamName('ve1Out')
ve1.bitstream.link(ve1Out.input)

ve2Out = pipeline.createXLinkOut()
ve2Out.setStreamName('ve2Out')
ve2.bitstream.link(ve2Out.input)

ve3Out = pipeline.createXLinkOut()
ve3Out.setStreamName('ve3Out')
ve3.bitstream.link(ve3Out.input)

xout_color = pipeline.createXLinkOut()
xout_color.setStreamName("color")
colorCam.video.link(xout_color.input)

ve4Out = pipeline.createXLinkOut()
ve4Out.setStreamName('ve4Out')
ve4.bitstream.link(ve4Out.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as dev:
    # Start pipeline
    dev.startPipeline()

    # Output queues will be used to get the encoded data from the outputs defined above
    outQ1 = dev.getOutputQueue(name='ve1Out', maxSize=30, blocking=True)
    outQ2 = dev.getOutputQueue(name='ve2Out', maxSize=30, blocking=True)
    outQ3 = dev.getOutputQueue(name='ve3Out', maxSize=30, blocking=True)
    outQ4 = dev.getOutputQueue(name='ve4Out', maxSize=30, blocking=True)
    q_rgb = dev.getOutputQueue(name="color", maxSize=4, blocking=True)

    # The .h264 / .h265 files are raw stream files (not playable yet)
    with open(new_loc+'mono1.h264', 'wb') as file_mono1_h264, open(new_loc+'color.h265', 'wb') as file_color_h265, open(new_loc+'mono2.h264', 'wb') as file_mono2_h264, open(new_loc+'depth.h264', 'wb') as file_depth_h264:
        print("Press 'Q' to stop encoding...")
        while True:
            in_rgb = q_rgb.get()
            shape = (in_rgb.getHeight()*3//2 , in_rgb.getWidth())
            frame_rgb = cv2.cvtColor(in_rgb.getData().reshape(shape), cv2.COLOR_YUV2BGR_NV12)
            frame_rgb=cv2.resize(frame_rgb,(640,360))
            cv2.imshow("bgr", frame_rgb)
            if cv2.waitKey(1) == ord('q'):
                break
            try:
                # Empty each queue
                while outQ1.has():
                    outQ1.get().getData().tofile(file_mono1_h264)

                while outQ2.has():
                    outQ2.get().getData().tofile(file_color_h265)

                while outQ3.has():
                    outQ3.get().getData().tofile(file_mono2_h264)

                while outQ4.has():
                    outQ4.get().getData().tofile(file_depth_h264)
            except KeyboardInterrupt:
                # Keyboard interrupt (Ctrl + C) detected
                break

    #print("To view the encoded data, convert the stream file (.qqh264/.h265) into a video file (.mp4), using commands below:")
    cmd = "ffmpeg -framerate 30 -i {} -c copy {}"

    names = ['mono1', 'mono2', 'color', 'depth']
    for name in names:
        if name == 'color':
            cmd_run = cmd.format('Videos/' + exercise + '/v' + version + '_' + exercise + '_' + name + '.h265',
                                 'Videos/' + exercise + '/v' + version + '_' + exercise + '_' + name + '.mp4')
            rm_file = 'Videos/' + exercise + '/v' + version + '_' + exercise + '_' + name + '.h265'
        else:
            cmd_run = cmd.format('Videos/' + exercise + '/v' + version + '_' + exercise + '_'+name+'.h264',
                    'Videos/' + exercise + '/v' + version + '_' + exercise + '_'+name+'.mp4')

            rm_file = 'Videos/' + exercise + '/v' + version + '_' + exercise + '_' + name + '.h264'
        #os.system(cmd_run)
        os.system('cmd /c ' + '"' + cmd_run + '"')

        os.remove(rm_file)