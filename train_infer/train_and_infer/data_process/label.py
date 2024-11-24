import cv2
import os
import json


# Initialize lists to store the points
white_points, red_points, green_points = [], [], []

input_path = './test_set_tmp'
output_path = './label_data/'
output_path_2 = './label_data_form/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path_2, exist_ok=True)

# 读取图像文件并过滤掉无效的文件名
images = [f for f in os.listdir(input_path) if f.endswith('.rgb.jpg')]
# images = os.listdir(input_path)
# sort the images according to the image id
if len(images) > 0:
    images.sort(key=lambda x: int(x.split('.')[0]))

print(images)
# find the existed frame id in the json name in the output folder, then remove corresponding images
for file in os.listdir(output_path):
    if file.endswith(".json"):
        frame_id = int(file.split('.')[0])
        if str(frame_id)+'.rgb.jpg' in images:
            images.remove(str(frame_id)+'.rgb.jpg')

print(images)
# remove image_file which not in images in the output_path and output_path_2
for file in os.listdir(output_path):
    if file.endswith(".rgb.jpg"):
        if file not in images:
            os.remove(os.path.join(output_path, file))

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:  # Left click for blue points
        green_points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Draw a blue circle
        cv2.imshow('image', img)
    elif event == cv2.EVENT_LBUTTONDOWN:  # Right click for red points
        red_points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # Draw a red circle
        cv2.imshow('image', img)
    elif event == cv2.EVENT_MBUTTONDOWN:
        white_points.append((x, y))
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)  # Draw a green circle
        cv2.imshow('image', img)
    cv2.imwrite(output_path+params, img)
    # save the original image to the output_path_2
    cv2.imwrite(output_path_2+params, img)

for image_name in images:
    img_path = os.path.join(input_path, image_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image: {image_name}")
        continue

    white_points, red_points, green_points = [], [], []  # Reset points for the new image
    # show the frame id on the image
    cv2.putText(img, image_name.split('.')[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event, image_name)
    key = cv2.waitKey(0)

    #只标注27个
    if key == 27:  # Press 'ESC' to exit the loop
        break

    # Merge close points
    keypoints_white_merged = white_points
    keypoints_red_merged = red_points
    keypoints_green_merged = green_points

    white_location = [[0,0,0] for i in range(len(white_points))]
    red_location = [[0,0,0] for i in range(len(red_points))]
    green_location = [[0,0,0] for i in range(len(green_points))]

    # Save the keypoints in the JSON format
    keypoints_data = {
        "objects": [{
            "class": "gecko",
            "keypoints": [
                {"name": "head", "location": white_location, "projected_location": keypoints_white_merged},
                {"name": "joint", "location": green_location, "projected_location": keypoints_green_merged},
                {"name": "box", "location": red_location, "projected_location": keypoints_red_merged}
            ]
        }]
    }


    print(keypoints_data)
    keypoints_json = json.dumps(keypoints_data, indent=4)
    # save the JSON file in ./label_data/{same file name}_keypoints.json
    json_path = output_path + image_name.replace('.rgb.jpg', '.json')
    with open(json_path, 'w') as f:
        f.write(keypoints_json)
    # save the json file in the output_path_2
    json_path_2 = output_path_2 + image_name.replace('.rgb.jpg', '.json')
    with open(json_path_2, 'w') as f:
        f.write(keypoints_json)

    # # press the space key to move to the next image
    # if key == 32:
    #     continue

cv2.destroyAllWindows()
