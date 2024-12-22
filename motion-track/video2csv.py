import os
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from ruamel.yaml import YAML
import dream
import csv


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


def network_inference(args):
    assert os.path.exists(
        args.input_params_path), f'Expected input_params_path "{args.input_params_path}" to exist, but it does not.'

    input_config_path = args.input_config_path or os.path.splitext(args.input_params_path)[0] + ".yaml"
    assert os.path.exists(
        input_config_path), f'Expected input_config_path "{input_config_path}" to exist, but it does not.'

    assert os.path.exists(args.image_path), f'Expected image_path "{args.image_path}" to exist, but it does not.'

    data_parser = YAML(typ="safe")
    with open(input_config_path, "r") as f:
        network_config = data_parser.load(f)

    dream_network = dream.create_network_from_config_data(network_config)
    dream_network.model.load_state_dict(torch.load(args.input_params_path))
    dream_network.enable_evaluation()

    capture = cv2.VideoCapture(args.image_path)

    csv_file_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['frame_id', 'head', 'center', 'forward', 'box_1', 'box_2', 'box_3', 'box_4', 'box_5', 'box_6',
                            'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])

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
            # calculate center and draw it with yellow color
            center = np.mean(points, axis=0)
            center = tuple(center.astype(int))

            # import pdb;pdb.set_trace()
            if len(head_data) == 0: break
            head = head_data[0]
            forward = head - center

            print(f"head: {head}, center: {center}, forward: {forward}")
            print(f"box: {tmp_box_data}, joint: {tmp_joint_data}")

            # Prepare data for CSV
            head = tuple(head)
            forward = tuple(forward)
            boxes = tmp_box_data
            joints = tmp_joint_data

            # Ensure we have 6 boxes and 6 joints
            while len(boxes) < 6:
                boxes.append(('', ''))
            while len(joints) < 6:
                joints.append(('', ''))

            # Write data to CSV, add frame_id into the first column
            csv_row = [frame_index, head, center, forward] + boxes + joints
            csvwriter.writerow(csv_row)

            tmp_image = np.array(keypoints_wrtNetInput_overlay)
            # draw center with yellow color
            cv2.circle(tmp_image, center, 5, (0, 255, 255), -1)
            tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Preview", tmp_image)

            # if cv2.waitKey(1) & 0xFF == ord("d"):
            #     break

    # capture.release()
    # cv2.destroyAllWindows()


class Arguments:
    def __init__(self):
        self.input_params_path = "./best_network_2.pth"
        self.image_path = "./video.mov" # your video path
        self.input_config_path = False
        self.network_config = None
        self.image_preproc_override = False
        self.keypoints_path = False
        self.save_dir = "save_data/video2csv"
        self.save_name = "test"


if __name__ == "__main__":
    args = Arguments()
    folder = "./original/"
    id = 0
    for video in [os.path.join(folder, video) for video in os.listdir(folder)]:
        id += 1
        args.image_path = video
        args.save_name = f"{id}.csv"
        network_inference(args)
    
    network_inference(args)