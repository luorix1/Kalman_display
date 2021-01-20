import cv2
import numpy as np
import time

#Open test data txt file
#file = open("data.txt", "r")
file = open("data_xy.txt", "r")

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('tree.avi')
count = 0
x_pos = 100
y_pos = 100
a_x = 0
a_y = 0
frames = 60

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  start = time.time()
  # Capture frame-by-frame
  ret, frame = cap.read()
  #gray =
  line = file.readline()
  if line:
      line = line.strip()
      a_x = int(line.split('	')[0])
      a_y = int(line.split('	')[1])
      scale_percent = int(line.split('	')[2])

  if ret == True:
    # Display the resulting frame
    resized = frame
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    #resize video frame for 5 frames
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('Frame',resized)

    #cv2.imshow('Frame', frame)
    
    #acceleration in m/s**2
    #x_pos = x_pos + int((a_x / frames) * (count - 50))
    #y_pos = y_pos + int((a_y / frames) * (count - 50))

    x_pos = x_pos + (a_x)
    y_pos = y_pos + (a_y)

    cv2.moveWindow('Frame', x_pos, y_pos)

    count = count + 1

    print(time.time() - start)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

#Close txt file
file.close()