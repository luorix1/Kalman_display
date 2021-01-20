import cv2
import time

def realtimeCV(cap, new):
    global x_pos
    global y_pos
    global a_x
    global a_y
    global scale_percent

    print("hello\n")

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

        start = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        line = new.readline()

        if line:
            line = line.strip()
            a_x = int(line.split('    ')[0])
            a_y = int(line.split('    ')[1])
            scale_percent = int(line.split('    ')[2])

        else:
            a_x = 0
            a_y = 0
            scale_percent = 100
            line = new.readline()

        if ret:
            # Display the resulting frame
            resized = frame
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize video frame
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame', resized)

            # cv2.imshow('Frame', frame)

            # acceleration in m/s**2
            x_pos = x_pos + (a_x)
            y_pos = y_pos + (a_y)

            cv2.moveWindow('Frame', x_pos, y_pos)

            print(time.time() - start)