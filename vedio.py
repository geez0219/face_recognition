import cv2
import numpy as np
import pdb
from face_recognition import FaceRecognition
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
app = FaceRecognition()
app.prepare(ctx_id=0, det_size=(640, 640))
app.read_facebase("./img/targets")
cap = cv2.VideoCapture('test.MOV')
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  frame2 = app.draw_on(frame)
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame2)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()
