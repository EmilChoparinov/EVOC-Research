"""An example on how to remote control a physical modular robot."""

from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1

from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_physical import Config, UUIDKey
from revolve2.modular_robot_physical.remote import run_remote

from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
import numpy as np
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, CpgNetworkStructure
import math
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)
record_flag = True


import cv2
import os
import time
import multiprocessing
import dream
import cv2
import torch
import numpy as np
from ruamel.yaml import YAML
from PIL import Image as PILImage
import json
from sklearn.cluster import KMeans


def record(stop_event, time_stamp):
    # cap = cv2.VideoCapture('rtsp://admin:123456@192.168.1.216/H264?ch=1&subtype=0')
    # cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.181:554/cam/realmonitor?channel=1&subtype=0')
    # cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.183:554/cam/realmonitor?channel=1&subtype=0')
    # cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.198:554/cam/realmonitor?channel=1&subtype=0')
    cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.199:554/cam/realmonitor?channel=1&subtype=0')
    # cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.200:554/cam/realmonitor?channel=1&subtype=0')
    # cap = cv2.VideoCapture('rtsp://admin:Robocam_0@10.15.1.201:554/cam/realmonitor?channel=1&subtype=0')

    # save the video file
    # check if the output path exists or not

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (400, 400)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = 'evaluate_video_' + time_stamp + '.mp4'
    # out = cv2.VideoWriter('./'+output_path+'/'+video_name, -1, 20.0, (720,720))
    out = cv2.VideoWriter('/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/'+video_name, fourcc, 20.0, size)
    # global record_flag
    while not stop_event.is_set():
        #print('About to start the Read command')
        ret, frame = cap.read()
        frame = frame[80:1200, 500:1620]
        frame = cv2.resize(frame, size)
        # cv2.imshow("Capturing",frame)
        frame = cv2.resize(frame, size)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def make_body() -> (
    tuple[BodyV2, tuple[ActiveHinge, ActiveHinge, ActiveHinge, ActiveHinge]]
):
    """
    Create a body for the robot.

    :returns: The created body and a tuple of all ActiveHinge objects for mapping later on.
    """
    # A modular robot body follows a 'tree' structure.
    # The 'Body' class automatically creates a center 'core'.
    # From here, other modular can be attached.
    # Modules can be attached in a rotated fashion.
    # This can be any angle, although the original design takes into account only multiples of 90 degrees.
    body = BodyV1()
    body.core_v1.left = ActiveHingeV1(RightAngles.DEG_0)
    body.core_v1.left.attachment = ActiveHingeV1(RightAngles.DEG_0)
    body.core_v1.left.attachment.attachment = BrickV1(RightAngles.DEG_0)
    body.core_v1.right = ActiveHingeV1(RightAngles.DEG_0)
    body.core_v1.right.attachment = ActiveHingeV1(RightAngles.DEG_0)
    body.core_v1.right.attachment.attachment = BrickV1(RightAngles.DEG_0)

    """Here we collect all ActiveHinges, to map them later onto the physical robot."""
    active_hinges = (
        body.core_v1.left,
        body.core_v1.left.attachment,
        body.core_v1.right,
        body.core_v1.right.attachment,
    )
    return body, active_hinges

def gecko_v2() -> BodyV2:
    """
    Sample robot with new HW config.

    :returns: the robot
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = (
        BrickV2(0.0)
    )
    active_hiinges = (
        body.core_v2.right_face.bottom,
        body.core_v2.left_face.bottom,
        body.core_v2.back_face.bottom,
        body.core_v2.back_face.bottom.attachment.front,
        body.core_v2.back_face.bottom.attachment.front.attachment.left,
        body.core_v2.back_face.bottom.attachment.front.attachment.right,
    )
    return body, active_hiinges

def on_prepared() -> None:
    """Do things when the robot is prepared and ready to start the controller."""
    # print("Done. Press enter to start the brain.")
    # input()
    pass


def main(weights, bound) -> None:
    """Remote control a physical modular robot."""
    rng = make_rng_time_seed()
    # Create a modular robot, similar to what was done in the simulate_single_robot example. Of course, you can replace this with your own robot, such as one you have optimized using an evolutionary algorithm.
    # body, hinges = make_body()

    # check the number in weights exceeds the bound or not, if exceeds, set the number to the bound
    for i in range(len(weights)):
        if weights[i] > bound:
            weights[i] = bound
        elif weights[i] < -bound:
            weights[i] = -bound

    body, hinges = gecko_v2()
    # brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)

    (
        cpg_network_structure,
        output_mapping,
    ) = active_hinges_to_cpg_network_structure_neighbor(hinges)

    brain = BrainCpgNetworkStatic.uniform_from_params(
        # params=np.array([-3.45001554,  3.47552386, -1.1123694 , -2.5518197 ,  3.27122787,
        # 1.41621476,  3.33163976,  2.58448528,  0.68867087]),
        params=np.array(weights),
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.pi / 2.0,
        output_mapping=output_mapping,
    )
    robot = ModularRobot(body, brain)
    """
    Some important notes to understand:
    - Hinge mappings are specific to each robot, so they have to be created new for each type of body. 
    - The pin`s id`s can be found on th physical robots HAT.
    - The order of the pin`s is crucial for a correct translation into the physical robot.
    - Each ActiveHinge needs one corresponding pin to be able to move. 
    - If the mapping is faulty check the simulators behavior versus the physical behavior and adjust the mapping iteratively.
    
    For a concrete implementation look at the following example of mapping the robots`s hinges:
    """
    hinge_1, hinge_2, hinge_3, hinge_4, hinge_5, hinge_6 = hinges
    test = 31
    hinge_mapping = {
        # 0, 1, 2, 15, 16, 31
        UUIDKey(hinge_1): 15, # right face bottom
        UUIDKey(hinge_2): 16, # left face bottom
        UUIDKey(hinge_3): 2, # back face bottom
        UUIDKey(hinge_4): 0, # back face bottom attachment front
        UUIDKey(hinge_5): 31, # back face bottom attachment front attachment left
        UUIDKey(hinge_6): 1, # back face bottom attachment front attachment right
    }

    """
    A configuration consists of the follow parameters:
    - modular_robot: The ModularRobot object, exactly as you would use it in simulation.
    - hinge_mapping: This maps active hinges to GPIO pins on the physical modular robot core.
    - run_duration: How long to run the robot for in seconds.
    - control_frequency: Frequency at which to call the brain control functions in seconds. If you also ran the robot in simulation, this must match your setting there.
    - initial_hinge_positions: Initial positions for the active hinges. In Revolve2 the simulator defaults to 0.0.
    - inverse_servos: Sometimes servos on the physical robot are mounted backwards by accident. Here you inverse specific servos in software. Example: {13: True} would inverse the servo connected to GPIO pin 13.
    """
    config = Config(
        modular_robot=robot,
        hinge_mapping=hinge_mapping,
        run_duration=60,
        control_frequency=30,
        initial_hinge_positions={UUIDKey(active_hinge): 0.0 for active_hinge in hinges},
        inverse_servos={},
    )

    """
    Create a Remote for the physical modular robot.
    Make sure to target the correct hardware type and fill in the correct IP and credentials.
    The debug flag is turned on. If the remote complains it cannot keep up, turning off debugging might improve performance.
    """

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    weights_name = 'evaluate_weights_' + time_stamp + '.txt'
    # weights_path = '/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/'+weights_name
    # save the weights of the brain to txt file weights_path
    # np.savetxt(weights_path, brain._weight_matrix, fmt='%f')
    # create process to record video
    stop_event = multiprocessing.Event()
    # Create and start the recording process
    record_process = multiprocessing.Process(target=record, args=(stop_event, time_stamp))
    record_process.start()

    # run the robot
    run_remote(
        config=config,
        hostname="10.15.3.93",  # "Set the robot IP here.
        debug=True,
        on_prepared=on_prepared,
    )

    stop_event.set()
    record_process.join()

    video_name = 'evaluate_video_' + time_stamp + '.mp4'
    video_path = '/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/'+video_name
    return video_path

def compute_distance_on_videos(args, video_path):
    # use dream to detect the first and last valid frame with head keypoint and compute the distance
    assert os.path.exists(
        args.input_params_path
    ), 'Expected input_params_path "{}" to exist, but it does not.'.format(args.input_params_path)
    if args.input_config_path:
        input_config_path = args.input_config_path
    else:
        # Use params filepath to infer the config filepath
        input_config_path = os.path.splitext(args.input_params_path)[0] + ".yaml"
    data_parser = YAML(typ="safe")

    with open(input_config_path, "r") as f:
        network_config = data_parser.load(f)

    # Load network
    dream_network = dream.create_network_from_config_data(network_config)
    dream_network.model.load_state_dict(torch.load(args.input_params_path))
    dream_network.enable_evaluation()
    capture = cv2.VideoCapture(video_path)
    first_head_location, last_head_location, first_center = None, None, None
    box_locations = []
    joint_locations = []
    skip_frames = 5
    index = -1
    size = (400, 400)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    video_name = 'infered_video' + time_stamp + '.mp4'
    out = cv2.VideoWriter('/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/'+video_name, fourcc, 20.0, size)
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        index += 1
        if index % skip_frames != 0:
            continue

        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb_OrigInput_asPilImage = PILImage.fromarray(color_converted)
        if args.image_preproc_override:
            image_preprocessing = args.image_preproc_override
        else:
            image_preprocessing = dream_network.image_preprocessing()
        detection_result = dream_network.keypoints_from_image(
            image_rgb_OrigInput_asPilImage,
            image_preprocessing_override=image_preprocessing,
            debug=True,
        )
        kp_coords_wrtNetInput_asArray = detection_result["detected_keypoints_net_input"]
        head_data, box_data, joint_data = [], [], []
        for name_index, input_keypoints_per_type in enumerate(kp_coords_wrtNetInput_asArray):
            if name_index == 0:
                head_data = input_keypoints_per_type
            elif name_index == 1:
                joint_data = input_keypoints_per_type
            elif name_index == 2:
                box_data = input_keypoints_per_type

        if len(head_data) > 0:
            if first_head_location is None:
                first_head_location = head_data[0]
            last_head_location = head_data[0]
            cv2.circle(frame, (int(head_data[0][0]), int(head_data[0][1])), 5, (0, 0, 255), -1)
        if len(box_data) > 0:
            for i in range(len(box_data)):
                cv2.circle(frame, (int(box_data[i][0]), int(box_data[i][1])), 2, (0, 255, 0), -1)
        if len(joint_data) > 0:
            for i in range(len(joint_data)):
                cv2.circle(frame, (int(joint_data[i][0]), int(joint_data[i][1])), 2, (0, 255, 0), -1)
        points = [point for point in head_data] + [point for point in box_data] + [point for point in joint_data]
        center = np.mean(points, axis=0)
        if first_center is None:
            first_center = center
        cv2.circle(frame, (int(first_center[0]), int(first_center[1])), 5, (255, 255, 255), -1)
        cv2.imshow("Capturing", frame)
        frame = cv2.resize(frame, size)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("d"):
            break
    capture.release()
    cv2.destroyAllWindows()
    # get the center of the box_locations
    box_locations = np.array(box_locations+joint_locations)

    # closest_point = np.mean(box_locations, axis=0)
    head_to_box = np.array(first_head_location) - np.array(first_center)

    # compute the euclidean distance between the first and last head location projection on the head_to_box direction
    # the distance could be minus if the head_to_box direction is opposite to the first and last head location
    # distance = np.dot(head_to_box, np.array(last_head_location) - np.array(first_head_location)) / np.linalg.norm(head_to_box)

    head_to_box_normalized = head_to_box / np.linalg.norm(head_to_box)
    # Compute the vector from the first to the last head location
    head_movement_vector = np.array(last_head_location) - np.array(first_head_location)
    projection = np.dot(head_movement_vector, head_to_box_normalized) * head_to_box_normalized
    projection_distance = np.dot(head_movement_vector, head_to_box_normalized)

    # If you need the signed distance to determine the direction (forward or backward along the head_to_box direction),
    # you can directly use the dot product without taking the absolute value
    signed_projection_distance = np.dot(head_movement_vector, head_to_box_normalized)

    print("first_head: ", first_head_location, " last_head: ", last_head_location, " distance: ", projection_distance, " center: ", first_center, " head_to_box: ", head_to_box)


    return projection_distance

    # compute the euclidean distance between the first and last head location
    # x1, y1, x2, y2 = first_head_location[0], first_head_location[1], last_head_location[0], last_head_location[1]
    # print("first_head: ", first_head_location, " last_head: ", last_head_location)
    # distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # return distance

class arguments1:
    def __init__(self):
        self.input_params_path = "/Users/xyh/Thesis/code/network/out/14/best_network_2.pth"
        self.image_path = None
        self.input_config_path = False
        self.network_config = None
        self.image_preproc_override = False
        self.keypoints_path = False
        self.save_dir = "evaluation_data/"
        self.save_name = False
        self.json_path = False

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

args = arguments1()
scale = 0.01
bound = 2.0
iter = 80

if __name__ == "__main__":
    # weight_tmp = np.random.uniform(-1.5, 1.5, 9)
    # weight_noise = weight_tmp + np.random.normal(0, 0.03, 9)
    # check the progress saved
    if os.path.exists('/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/saved_weights.json'):
        with open('/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/saved_weights.json', 'r') as f:
            data = json.load(f)
            data = json.loads(data)

            max_xy_dis_log = data['max_xy_dis_log']
            xy_dis_log = data['xy_dis_log']
            weights_pick_log = data['weights_pick_log'] if 'weights_pick_log' in data else []
            video_log = data['video_log'] if 'video_log' in data else []
            weights_log = data['weights_log'] if 'weights_log' in data else []
            weight_tmp = data['weights_pick_log'][-1] if 'weights_pick_log' in data else np.zeros(9) + np.random.normal(0, scale, 9)
    else:
        weight_tmp = np.zeros(9) + np.random.normal(0, scale, 9)
        max_xy_dis_log = []
        xy_dis_log = []
        weights_log = []
        video_log = []
        weights_pick_log = []
    weights_list = [
        weight_tmp
        # weight_noise,
    ]
    # weights_pick_log.append(weights_list)

    for i in range(iter):
        video_path_list = []
        xy_dis_list = []
        print("last weights picked: ", weights_pick_log[-1] if len(weights_pick_log) > 0 else "None")
        noise = np.random.normal(0, scale, 9)
        print("noise: ", noise)
        weights_list = [
            weights_pick_log[-1] + noise,
            weights_pick_log[-1] + np.random.normal(0, scale, 9),
            weights_pick_log[-1] + np.random.normal(0, scale, 9),
            weights_pick_log[-1] + np.random.normal(0, scale, 9),
            weights_pick_log[-1] + np.random.normal(0, scale, 9),
        ]
        print("current weights: ", weights_list)
        for weights in weights_list:
            # time.sleep(5)
            video_path = main(weights, bound)
            video_path_list.append(video_path)
            weights_log.append(weights)
            time.sleep(5)
        for video_path in video_path_list:
            xy_dis = compute_distance_on_videos(args, video_path)
            xy_dis_list.append(xy_dis)
        # find the bigger distance in xy_dis_list and its index
        print("xy_dis_list: ", xy_dis_list)
        max_xy_dis = max(xy_dis_list)
        max_index = xy_dis_list.index(max_xy_dis)
        print("max_xy_dis: ", max_xy_dis)
        print("max_xy_dis weight: ", weights_list[max_index])
        xy_dis_log.append(max_xy_dis)
        video_log.append(video_path_list[0])
        if len(max_xy_dis_log) == 0:
            max_xy_dis_log.append(max_xy_dis)
            weights_pick_log.append(weights_list[0])
        # compare the max_xy_dis with max_xy_dis_log[-1]
        elif len(max_xy_dis_log) > 0 and max_xy_dis > max_xy_dis_log[-1]:
            # keep the weights of the bigger distance
            max_xy_dis_log.append(max_xy_dis)
            weights_pick_log.append(weights_list[0])

        # else:
        #     # ignore the weights of the bigger distance
        #     weights_list = [
        #         # weights_list[0],
        #         # add gaussian noise to the weights of the smaller distance
        #         weights_list[0] + np.random.normal(0, scale, 9),
        #     ]
        #     # weights_pick_log.append(weights_list)

        print("iter: ", i)
        print("@@max_xy_dis_log: ", max_xy_dis_log)
        print("weights_pick_log: ", weights_pick_log)

        print("xy_dis_log: ", xy_dis_log)
        print("weights_log: ", weights_log)
        # print("video_log: ", video_log)

        # save the progress
        data = {
            'weights_pick_log': weights_pick_log,
            'max_xy_dis_log': max_xy_dis_log,
            'xy_dis_log': xy_dis_log,
            'video_log': video_log,
            'weights_log': weights_log,
        }
        # import pdb; pdb.set_trace()
        dumped = json.dumps(data, cls=NumpyEncoder)
        with open('/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_50_iter/saved_weights.json', 'w') as f:
            json.dump(dumped, f)
        # time.sleep(5)
    # display a diagram of the max_xy_dis_log
    import matplotlib.pyplot as plt
    plt.plot(max_xy_dis_log)
    # main()
