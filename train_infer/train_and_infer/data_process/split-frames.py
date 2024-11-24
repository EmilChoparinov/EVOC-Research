import cv2
import os

# Make sure the 'raw' directory exists
output_folder = './test_set_tmp'
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(output_folder):
    print("Failed to create output folder.")
else:
    print(f"Output folder '{output_folder}' created successfully.")

#每隔32帧提取一次
skip_frame = 32


videos_path = './video/'

# find the bigest frame id in the output folder
frame_id = 0
for file in os.listdir(output_folder):
    if file.endswith(".rgb.jpg"):
        #提取文件名中的帧编号并更新frame_id
        frame_id = max(frame_id, int(file.split('.')[0]))

for video_name in os.listdir(videos_path):
    print(f'Processing video {video_name}')
    cap = cv2.VideoCapture(videos_path+video_name)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Read until video is completed
    tmp_frame_id = 0
    while cap.isOpened():
        if frame_id==1000:
            break
        ret, frame = cap.read()
        if ret:
            # Save frame as JPEG file
            if tmp_frame_id % skip_frame == 0:
                cv2.imwrite(f'{output_folder}/{frame_id}.rgb.jpg', frame)
                print(f'Saved frame {frame_id}')
                frame_id += 1
            tmp_frame_id += 1
        else:
            break
    cap.release()
print('Video processing complete.')
