import os
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from ruamel.yaml import YAML
import dream
import csv
import json


def generate_belief_map_visualizations(belief_maps, keypoint_projs_detected, keypoint_projs_gt=None):
    belief_map_images = dream.image_proc.images_from_belief_maps(belief_maps, normalization_method=6)
    belief_map_images_kp = []
    for kp in range(len(keypoint_projs_detected)):
        if keypoint_projs_gt:
            keypoint_projs = [keypoint_projs_gt[kp], keypoint_projs_detected[kp]]
            color = ["green", "red"]
        else:
            keypoint_projs = [keypoint_projs_detected[kp]]
            color = "red"
        belief_map_image_kp = dream.image_proc.overlay_points_on_image(
            belief_map_images[kp],
            keypoint_projs,
            annotation_color_dot=color,
            annotation_color_text=color,
            point_diameter=4,
        )
        belief_map_images_kp.append(belief_map_image_kp)
    n_cols = int(np.ceil(len(keypoint_projs_detected) / 2.0))
    belief_maps_kp_mosaic = dream.image_proc.mosaic_images(
        belief_map_images_kp,
        rows=2,
        cols=n_cols,
        inner_padding_px=10,
        fill_color_rgb=(0, 0, 0),
    )
    return belief_maps_kp_mosaic


def load_weights_data(folder_path):
    weights_file = os.path.join(folder_path, 'saved_weights.json')
    if os.path.exists(weights_file):
        with open(weights_file, 'r') as f:
            data = json.load(f)
            data = json.loads(data)  # Parse the nested JSON string
        return data
    return None


def find_params_for_video(weights_data, video_name):
    if weights_data and 'video_log' in weights_data and 'weights_log' in weights_data:
        for i, video_path in enumerate(weights_data['video_log']):
            if video_name in video_path:
                return weights_data['weights_log'][i]
    return None


def process_video(args, video_path, save_dir, weights_data):
    assert os.path.exists(
        args.input_params_path), f'Expected input_params_path "{args.input_params_path}" to exist, but it does not.'

    input_config_path = args.input_config_path or os.path.splitext(args.input_params_path)[0] + ".yaml"
    assert os.path.exists(
        input_config_path), f'Expected input_config_path "{input_config_path}" to exist, but it does not.'

    assert os.path.exists(video_path), f'Expected video_path "{video_path}" to exist, but it does not.'

    data_parser = YAML(typ="safe")
    with open(input_config_path, "r") as f:
        network_config = data_parser.load(f)

    dream_network = dream.create_network_from_config_data(network_config)
    dream_network.model.load_state_dict(torch.load(args.input_params_path))
    dream_network.enable_evaluation()

    capture = cv2.VideoCapture(video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_file_path = os.path.join(save_dir, f'{video_name}.csv')
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    params = find_params_for_video(weights_data, video_name)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['frame_id', 'head', 'center', 'forward', 'box_1', 'box_2', 'box_3', 'box_4', 'box_5', 'box_6',
                  'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        if params:
            header.append('parameters')
        csvwriter.writerow(header)

        frame_index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_index += 1
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb_OrigInput_asPilImage = PILImage.fromarray(color_converted)

            image_preprocessing = args.image_preproc_override or dream_network.image_preprocessing()

            detection_result = dream_network.keypoints_from_image(
                image_rgb_OrigInput_asPilImage,
                image_preprocessing_override=image_preprocessing,
                debug=True,
            )

            kp_coords_wrtNetInput_asArray = detection_result["detected_keypoints_net_input"]
            image_rgb_NetInput_asPilImage = detection_result["image_rgb_net_input"]

            head_data, box_data, joint_data = None, None, None
            for name_index, input_keypoints_per_type in enumerate(kp_coords_wrtNetInput_asArray):
                if name_index == 0:
                    head_data = input_keypoints_per_type
                elif name_index == 1:
                    box_data = input_keypoints_per_type
                else:
                    joint_data = input_keypoints_per_type

                keypoints_wrtNetInput_overlay = dream.image_proc.overlay_points_on_image(
                    image_rgb_NetInput_asPilImage,
                    input_keypoints_per_type,
                    [dream_network.friendly_keypoint_names[name_index]] * len(input_keypoints_per_type),
                    annotation_color_dot=["red", "blue", "green"][name_index],
                    annotation_color_text="white",
                )
                image_rgb_NetInput_asPilImage = keypoints_wrtNetInput_overlay

            tmp_box_data = [(box[0], box[1]) for box in box_data]
            tmp_joint_data = [(joint[0], joint[1]) for joint in joint_data]

            points = tmp_box_data + tmp_joint_data
            center = np.mean(points, axis=0)
            center = tuple(center.astype(int))
            if len(head_data) == 0:
                continue
            head = head_data[0]
            forward = head - center

            # print(f"Processing frame {frame_index} of {video_name}")
            # print(f"head: {head}, center: {center}, forward: {forward}")
            # print(f"box: {tmp_box_data}, joint: {tmp_joint_data}")

            head = tuple(head)
            forward = tuple(forward)
            boxes = tmp_box_data
            joints = tmp_joint_data

            while len(boxes) < 6:
                boxes.append(('', ''))
            while len(joints) < 6:
                joints.append(('', ''))

            csv_row = [frame_index, head, center, forward] + boxes + joints
            if params:
                csv_row.append(params)
            csvwriter.writerow(csv_row)

    capture.release()
    print(f"Finished processing {video_name}")


def process_folders(args, root_folder):
    folder_list = os.listdir(root_folder)
    # sort the folders by name
    folder_list.sort()
    for folder_name in folder_list:
        print(f"Processing folder {folder_name}")
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            save_dir = os.path.join("./save_data", folder_name)
            weights_data = load_weights_data(folder_path)
            file_list = os.listdir(folder_path)
            # sort the files by name
            file_list.sort()
            for video_file in file_list:
                if video_file.endswith(('.mp4', '.avi', '.mov')) and video_file.startswith('evaluate'):
                    video_path = os.path.join(folder_path, video_file)
                    print(f"Processing video: {video_path}")
                    process_video(args, video_path, save_dir, weights_data)
                    print(f"Finished processing video {video_file}")
        print(f"Finished processing folder {folder_name}")


class Arguments:
    def __init__(self):
        self.input_params_path = "../train/out/14/best_network_2.pth"
        self.input_config_path = False
        self.network_config = None
        self.image_preproc_override = False
        self.keypoints_path = False


if __name__ == "__main__":
    args = Arguments()
    root_folder = "./experiment_data/"
    process_folders(args, root_folder)