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

# size = (400, 400)

def record(stop_event, time_stamp):
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
    size = (400, 400)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)

    output_path = 'evaluate_output_params'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_name = 'evaluate_video_' + time_stamp + '.mp4'
    # out = cv2.VideoWriter('./'+output_path+'/'+video_name, -1, 20.0, (720,720))
    out = cv2.VideoWriter('/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_output_params/'+video_name, fourcc, 20.0, size)
    # global record_flag
    while not stop_event.is_set():
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


def main(weights) -> None:
    """Remote control a physical modular robot."""
    rng = make_rng_time_seed()
    # Create a modular robot, similar to what was done in the simulate_single_robot example. Of course, you can replace this with your own robot, such as one you have optimized using an evolutionary algorithm.
    # body, hinges = make_body()
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
        # 0, 1, 15, 16, 18, 31
        UUIDKey(hinge_1): 1, # right face bottom
        UUIDKey(hinge_2): 1, # left face bottom
        UUIDKey(hinge_3): 1, # back face bottom
        UUIDKey(hinge_4): 1, # back face bottom attachment front
        UUIDKey(hinge_5): 1, # back face bottom attachment front attachment left
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
        # inverse_servos={15: True, 16: True, 2: True, 0: True, 31: True, 1: True},
    )

    """
    Create a Remote for the physical modular robot.
    Make sure to target the correct hardware type and fill in the correct IP and credentials.
    The debug flag is turned on. If the remote complains it cannot keep up, turning off debugging might improve performance.
    """
    print("Initializing robot..")
    print("Starting video recording..")
    print("weights of the brain: ", brain._weight_matrix)
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    weights_name = 'evaluate_weights_' + time_stamp + '.txt'
    weights_path = '/Users/xyh/Thesis/code/revolve2/examples/physical_robot_remote/evaluate_output_params/'+weights_name
    # save the weights of the brain to txt file weights_path
    np.savetxt(weights_path, brain._weight_matrix, fmt='%f')
    # create process to record video
    stop_event = multiprocessing.Event()
    # Create and start the recording process
    record_process = multiprocessing.Process(target=record, args=(stop_event, time_stamp))
    record_process.start()

    # run the robot
    run_remote(
        config=config,
        hostname="10.15.3.47",  # "Set the robot IP here.
        debug=True,
        on_prepared=on_prepared,
    )

    print("program ended")
    stop_event.set()
    record_process.join()




if __name__ == "__main__":
    weight_tmp = np.random.uniform(-1.5, 1.5, 9)
    weight_noise = weight_tmp + np.random.normal(0, 0.03, 9)
    # weights_list = [
    #     [-1.11810673, 0.46544229, 0.14996644, 2., -1.30786652,
    #      0.97685073, 0.45806286, -0.73357919, -0.22266606]
    # ]
    weights_list = [
        [0.4218761046756034, 0.25186484935411746, 0.7969127263214236, 0.028642818147120663, -0.15047066237168719, -0.5757953440429397, 0.07351398888547789, -0.1064885980025415, -0.4643283117920747]
    ]
    iter = 1
    for i in range(iter):
        for weights in weights_list:
            main(weights)
            # time.sleep(10)
    # main()
