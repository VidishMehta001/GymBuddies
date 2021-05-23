import blazepose_extract as bpe

obj=bpe()

# For webcam which will print the X,Y of landmarks
obj.video_pose()

# For video which will print the X,Y of landmarks
obj.video_pose("filename")

# For image which will return the X,Y of landmarks
x,y = obj.video_pose(["filename"])
