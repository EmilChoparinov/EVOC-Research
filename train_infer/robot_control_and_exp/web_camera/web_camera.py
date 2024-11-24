import cv2
import os
import time
#print("Before URL")
# cap = cv2.VideoCapture('rtsp://admin:123456@192.168.1.216/H264?ch=1&subtype=0')
# cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.181:554/cam/realmonitor?channel=1&subtype=0')
# cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.183:554/cam/realmonitor?channel=1&subtype=0')
# cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.198:554/cam/realmonitor?channel=1&subtype=0')
cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.199:554/cam/realmonitor?channel=1&subtype=0')
# cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.200:554/cam/realmonitor?channel=1&subtype=0')
# cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.201:554/cam/realmonitor?channel=1&subtype=0')
#print("After URL")

# save the video file
# check if the output path exists or not

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (920, 920)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)

output_path = 'output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

video_name = 'video_' + time.strftime("%Y%m%d-%H%M%S") + '.mp4'
# out = cv2.VideoWriter('./'+output_path+'/'+video_name, -1, 20.0, (720,720))
out = cv2.VideoWriter('./'+output_path+'/'+video_name, fourcc, 20.0, size)
while True:

    #print('About to start the Read command')
    ret, frame = cap.read()

    # zoom into the center of the image like half of the image
    frame = frame[80:1200, 500:1620]
    # resize the image to 720x720
    frame = cv2.resize(frame, size)
    #print('About to show frame of Video.')
    cv2.imshow("Capturing",frame)
    # save the video frame to video file
    out.write(frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()