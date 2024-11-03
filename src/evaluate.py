from revolve2.modular_robot import ModularRobot
from config import PhysMap, target_angle_per_run, target_dist_per_run

from revolve2.modular_robot_simulation import ModularRobotSimulationState
from revolve2.modular_robot.body.v2 import ActiveHingeV2
from revolve2.modular_robot_simulation import SceneSimulationState

import pandas as pd
import numpy as np
import numpy.typing as npt
import math

from theory import coords_to_rad_vspace, polar_to_cartesian_2d, rel_op, cartesian_to_polar_nd

from typedef import population, simulated_behavior, genotype

_fvector = npt.NDArray[np.float_]
def project_nd(a: _fvector, b: _fvector) -> _fvector:
    return (np.dot(a, b) / np.dot(b, b)) * b

def get_pose_of(robot: ModularRobot, state: SceneSimulationState, part) -> tuple[float,float,float]:
    return state.get_modular_robot_simulation_state(robot).get_module_absolute_pose(part).position

def get_pose_x_delta(state0: ModularRobotSimulationState, stateN: ModularRobotSimulationState) -> float:
    """
    Calculate the different between the starting position and the final position
    
    Note: Its subtracting to produce a negative value because the robot just
          happend to spawn oriented towards the -x direction
    """
    return state0.get_pose().position.x - stateN.get_pose().position.x

def get_robot_euclidian_center(robot: ModularRobot, state: SceneSimulationState) -> tuple[float, float]:
    # Finding euclidian center routine    
    # Select all hinges, should be enough points to calculate
    hinges = robot.body.find_modules_of_type(ActiveHingeV2)
    
    # Get the pose function out and transform the position to a pandas dataframe
    f = state.get_modular_robot_simulation_state(robot).get_module_absolute_pose    
    xyzs = [f(hinge).position for hinge in hinges]
    pd_xys = pd.DataFrame([(xyz.x, xyz.y, xyz.z) for xyz in xyzs], columns=['x', 'y', 'z'])

    # Calculate the center by taking the mean of each component independently
    return (pd_xys['x'].mean(), pd_xys['y'].mean(), pd_xys['z'].mean())
    

def get_robot_euclidian_head(robot: ModularRobot, state: SceneSimulationState) -> tuple[float, float, float]:
    return state.get_modular_robot_simulation_state(robot).get_module_absolute_pose(robot.body.core_v2).position

def get_pose_theta_xy_delta(robot: ModularRobot, behavior: simulated_behavior, target: tuple[float, float]) -> float:
    """
    Calculate the angular difference between the final position and the target. 
    The target coordinate along the polar system has it's origin as the 
    euclidian center of the robot from the start of the simulation.
    """
    origin_x, origin_y,_ = get_robot_euclidian_head(robot, behavior[0])
    end_x, end_y,_ = get_robot_euclidian_head(robot, behavior[-1])

    # Make the end coordinates relative to the origin (in cartesian)
    relative_end_x, relative_end_y = rel_op(np.array([origin_x, origin_y]), 
                                            np.array([end_x, end_y])
    )
    
    _, angle = cartesian_to_polar_nd(
        np.array([relative_end_x, relative_end_y])
    )
    
    _, target_angle = target
    
    angle_from_target = abs(target_angle - angle[0])
    
    # Get the most acute angle to the target
    return min(angle_from_target, math.pi - target_angle)

def get_pose_projected_accuracy_on_target(robot: ModularRobot, behavior: simulated_behavior, target: tuple[float, float]) -> float:
    """
    Calculate the difference in magnitude between the end vector and the
    projected end vector over the target vector. The difference in magnitiude is
    the returned value between [0, 1] where 1 is exact match. 
    
    The key insight is we are adjusting the robots current position to be where
    its supposed to be if it had the right angle.
    """
    target_dist, target_angle = target
    
    origin_x, origin_y,_ = get_robot_euclidian_head(robot, behavior[0])
    end_x, end_y,_ = get_robot_euclidian_head(robot, behavior[-1])
    
    # Make the end coordinates relative to the origin (in cartesian)
    relative_end_x, relative_end_y = rel_op(
        np.array([origin_x, origin_y]),
        np.array([end_x, end_y])
    )
    
    proj = project_nd(
        np.array([relative_end_x, relative_end_y]),
        np.array(polar_to_cartesian_2d(target_dist, target_angle))
    )
    
    proj_mag = np.linalg.norm(proj)
    return proj_mag / target_dist

    # Find the relative magnititude (the distance) away from the origin
    relative_mag_end = np.linalg.norm(
        np.array([relative_end_x,relative_end_y])
    )
    
    # Use the magnitiude to construct the expected location IF the robot exactly
    # followed the correct angle
    expected_relative_x, expected_relative_y = polar_to_cartesian_2d(
        relative_mag_end, target_angle
    )
    
    # Project the actual position over the expected position
    proj_x, proj_y = project_nd(
        np.array([relative_end_x, relative_end_y]), 
        np.array([expected_relative_x, expected_relative_y])
    )
    
    # Get the magnititude of the projection 
    proj_mag = np.linalg.norm(np.array([proj_x, proj_y]))
    
    # returns [0,1] value representing the percent difference between the actual
    # and expected
    return proj_mag / relative_mag_end

def get_pose_z_avg(robot: ModularRobot, behavior: simulated_behavior) -> float:
    """
    Get the average in the z direction using the initial state as the origin
    """
    _,_,origin_z = get_robot_euclidian_head(robot, behavior[0])
    _,_,end_z = get_robot_euclidian_head(robot, behavior[0])
    
    return end_z - origin_z

def get_pose_z_airtime(robot: ModularRobot, behavior: simulated_behavior) -> float:
    """
    How much of the time is the head spending +2 cm above the resting position
    """
    head = robot.body.core_v2
    
    # Calculate the bound
    origin_z = behavior[0].get_modular_robot_simulation_state(robot).get_module_absolute_pose(head).position.z
    bound_z = origin_z + 0.02

    zs = [state.get_modular_robot_simulation_state(robot).get_module_absolute_pose(robot.body.core_v2).position.z 
        for state in behavior]
    
    time_correct = list(filter(lambda z: z > bound_z, zs))
    
    return len(time_correct) / len(zs)

def get_pose_cumulative_maximal_rotation(robot: ModularRobot, behavior: simulated_behavior) -> float:
    """
    The total amount of rotation performed by the robot in the simulation. Can go over
    one full 360 degree rotation. Will keep adding. Always looks at difference
    between start and current
    """
    # Create the origin to be the (0,0). We only care about rotation
    # in 2D space
    
    # origin_x, origin_y = (0,0)
    # start_x, start_y, _ = get_robot_euclidian_head(robot, behavior)
    
    pmap = PhysMap.map_with(robot.body)
    vorigin_head = get_pose_of(
        robot, behavior[0], robot.body.core_v2)
    
    vorigin_tail = get_pose_of(
        robot, behavior[0], pmap["tail"]["hinge"] 
    )
    
    # For all measurements, we add the total radians rotated from
    # the origin vector
    return sum([coords_to_rad_vspace(
        vorigin_head, vorigin_tail,
        get_pose_of(robot, state, robot.body.core_v2),
        get_pose_of(robot, state, pmap["tail"]["hinge"])
    ) for state in behavior])


def get_pose_maximal_rotation(robot: ModularRobot, behavior: simulated_behavior) -> float:
    """
    The total amount of rotation performed by the robot in the simulation. Can go over
    one full 360 degree rotation. Will keep adding
    """
    pmap = PhysMap.map_with(robot.body)
    vorigin_head = get_pose_of(
        robot, behavior[0], robot.body.core_v2)
    
    vorigin_tail = get_pose_of(
        robot, behavior[0], pmap["tail"]["hinge"] 
    )
    
    n = len(behavior)
    state_pairs = list(zip(range(0, n, 2), range(1, n, 2)))
    
    # For all measurements, we add the total radians rotated from
    # the previous measurement
    return sum([coords_to_rad_vspace(
        vorigin_head, vorigin_tail,
        get_pose_of(robot, behavior[idx_cur], robot.body.core_v2),
        get_pose_of(robot, behavior[idx_next], pmap["tail"]["hinge"])) 
     for idx_cur,idx_next in state_pairs])

def get_pose_maximal_rotation_filter_dir(robot: ModularRobot, behavior: simulated_behavior, go_left: bool) -> float:
    """
    The total amount of rotation performed by the robot in the simulation. Can go over
    one full 360 degree rotation. Will keep adding
    """
    pmap = PhysMap.map_with(robot.body)
    vorigin_head = get_pose_of(
        robot, behavior[0], robot.body.core_v2)
    
    vorigin_tail = get_pose_of(
        robot, behavior[0], pmap["tail"]["hinge"] 
    )
    
    n = len(behavior)
    state_pairs = list(zip(range(0, n, 2), range(1, n, 2)))
    
    # For all measurements, we add the total radians rotated from
    # the previous measurement
    rad_deltas = [
        coords_to_rad_vspace(
            vorigin_head, vorigin_tail,
            get_pose_of(robot, behavior[idx_cur], robot.body.core_v2),
            get_pose_of(robot, behavior[idx_next], pmap["tail"]["hinge"])
        ) 
     for idx_cur, idx_next in state_pairs]
    return sum(rad[0] for rad in rad_deltas if (go_left and rad[1] == "left") or (not go_left and rad[1] == "right"))

def evaluate_angle_with_projection_with_z_avg(
    robots: list[ModularRobot], behaviors: list[simulated_behavior], 
    run_idx: int, gen_idx: int
) -> npt.NDArray[np.float_]:
    target = (target_dist_per_run[run_idx-1] , target_angle_per_run[run_idx-1])

    def evalf(robot: ModularRobot, states: simulated_behavior):
        # return get_pose_cumulative_maximal_rotation(robot, states)
        return get_pose_maximal_rotation_filter_dir(robot, states, go_left=False)


        _,_,origin_z = get_robot_euclidian_head(robot, states[0])
        airtime = get_pose_z_airtime(robot, states)
        similarity = get_pose_projected_accuracy_on_target(
            robot, states, target)
        angle_diff = get_pose_theta_xy_delta(robot, states, target)
        
        # `angle_diff` is in radians. This value conveniently at 0.1 radians
        # (about 5 degrees) will produce the value 1. We clamp this to max
        # at one. Everything else gradually gives less score steeply
        # angle_metric = min(0.1 / angle_diff, 1)
        
        # return airtime + similarity + angle_metric
        # return airtime + angle_metric
        # return angle_metric

        # We gradually shift objectives between angling and walking forward again
        # over 20 generations, starting at generation 20. This artificially 
        # simulates the environment changing in perspective to the robot
        # ease = np.linspace(0, 1, 20)
        
        # alpha = ease[min(19, gen_idx-19)]
        # if gen_idx == 23: import pdb; pdb.set_trace()
        # if gen_idx < 20: alpha = 0
        # print(f"{similarity=}\n{angle_metric=}\n")
        # return (1 - alpha) * angle_metric + alpha * similarity
        # import pdb;pdb.set_trace()
        
        # ease = np.linspace(0, 1, 10)
        # alpha = ease[min(9,gen_idx)]
        # return (1 - alpha) * angle_metric + (alpha) * similarity
        
        
        # if angle_metric == 1:
        #     print(f"score:{angle_metric + similarity}")
        #     # extend with similarity once angle is met
        #     return angle_metric + similarity 
        # print(f"score:{angle_metric}")
        # return angle_metric


        if run_idx < 20: return angle_metric
        else: return similarity

        return 0.1* angle_metric + 0.9*similarity

    return np.array([
        evalf(robot, states) for robot, states in zip(robots, behaviors)
    ])



def evaluate_pose_x_delta(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
    """
    Perform evaluation over a list of robots. The incoming data is **assumed**
    to be ordered. I.E. the first index in the modular robot list has its 
    behavior recorded in the first index of the behavior list.
    
    Returns an array of ordered fitness values.
    """
    return np.array([
        # Get delta from first state to last state in simulation
        get_pose_x_delta(
            states[0].get_modular_robot_simulation_state(robot),
            states[-1].get_modular_robot_simulation_state(robot)
        ) for robot, states in zip(robots, behaviors)
    ])

def find_most_fit(fitnesses: npt.NDArray[np.float_], robots: list[ModularRobot], behaviors: list[simulated_behavior]):
    """
    Perform linear search for the robot with the highest fitness. The incoming
    parameters are **assumed** to be ordered.
    
    Returns tuple of best fitting robot with its behavior and fitness value
    """
    # Zip fitnesses with index, find the max fitness value and index 
    fittest_idx, fitness = max(enumerate(fitnesses), key=lambda x: x[1])
    return (robots[fittest_idx],behaviors[fittest_idx], fitness)
