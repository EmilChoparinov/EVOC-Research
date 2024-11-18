from revolve2.modular_robot import ModularRobot
from config import PhysMap, target_angle_per_run, target_dist_per_run

from revolve2.modular_robot_simulation import ModularRobotSimulationState
from revolve2.modular_robot.body.v2 import ActiveHingeV2
from revolve2.modular_robot_simulation import SceneSimulationState

import pandas as pd
import numpy as np
import numpy.typing as npt
import math
from scipy.stats import norm

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

def get_pose_nd_projected_mag(
    robot: ModularRobot, 
    behavior: simulated_behavior, 
    target: npt.NDArray[np.float_]
    ) -> float:
    origin_x, origin_y, _ = get_robot_euclidian_head(robot, behavior[0])
    end_x, end_y, _ = get_robot_euclidian_head(robot, behavior[-1])

    actual = rel_op(
        np.array([origin_x, origin_y]),
        np.array([end_x, end_y])
    )

    dot_product = np.dot(actual, target)
        
    return dot_product / (np.linalg.norm(target) ** 2)

def get_pose_nd_projected_normal_mag(
        robot: ModularRobot, 
        behavior: simulated_behavior, 
        target: npt.NDArray[np.float_]
    ) -> float:
        origin_x, origin_y, _ = get_robot_euclidian_head(robot, behavior[0])
        end_x, end_y, _ = get_robot_euclidian_head(robot, behavior[-1])

        actual = rel_op(
            np.array([origin_x, origin_y]),
            np.array([end_x, end_y])
        )

        target_norm = target / np.linalg.norm(target) 
        actual_norm = actual/ np.linalg.norm(actual)

        cos_theta = np.dot(actual_norm, target_norm)

        # Since both vectors are normalized before projection, the dot product
        # of the projection returns cos(theta). We must transform the following
        # cases of cos(theta) into a form that represents positive contribution
        # towards the angle of the target.
        # < 0 :-> point in opposite direction
        # > 0 :-> point in the same direction
        # = 0 :-> angles are perpendicular
        # The proposed solution is to do max(cos(theta), 0), since negative 
        # values mean the angle is obtuse in vector space.
        return max(cos_theta, 0)


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

def get_pose_muscle_effiency(robot: ModularRobot, behavior: simulated_behavior) -> float:
    """
    How efficient the hinge use was for this robot. Cumulative difference in hinge
    value passed through sigmoid(abs(x)).

    This function returns a penality value, so smaller is better!
    """
    n = len(behavior)
    state_pairs = list(zip(range(0,n,2), range(1,n,2)))

    def eval_effiency(s0: ModularRobotSimulationState, sN: ModularRobotSimulationState):
        hinges = robot.body.find_modules_of_type(ActiveHingeV2)
        
        s0_hinges = [s0.get_hinge_position(h) for h in hinges]
        sN_hinges = [sN.get_hinge_position(h) for h in hinges]

        # Calculates the overall effiency of the robot in this specific state
        x = sum([
            math.pow(abs(s0_pos - sN_pos),3)
            for s0_pos, sN_pos in zip(s0_hinges, sN_hinges)
        ])
        return x


    # The penality is fast rates of change on the hinges, AKA more energy
    return sum([eval_effiency(
            behavior[cur_i].get_modular_robot_simulation_state(robot), 
            behavior[next_i].get_modular_robot_simulation_state(robot)
        ) for cur_i, next_i in state_pairs])



def get_pose_cumulative_maximal_rotation(robot: ModularRobot, behavior: simulated_behavior) -> float:
    """
    The total amount of rotation performed by the robot in the simulation. Can go over
    one full 360 degree rotation. Will keep adding. Always looks at difference
    between start and current
    """
    pmap = PhysMap.map_with(robot.body)
    vorigin_head = get_pose_of(
        robot, behavior[0], robot.body.core_v2)
    
    vorigin_tail = get_pose_of(
        robot, behavior[0], pmap["tail"]["box"] 
    )
    
    # For all measurements, we add the total radians rotated from
    # the origin vector
    return sum([coords_to_rad_vspace(
        vorigin_head, vorigin_tail,
        get_pose_of(robot, state, robot.body.core_v2),
        get_pose_of(robot, state, pmap["tail"]["box"])
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
        robot, behavior[0], pmap["tail"]["box"] 
    )
    
    n = len(behavior)
    state_pairs = list(zip(range(0, n, 2), range(1, n, 2)))
    
    # For all measurements, we add the total radians rotated from
    # the previous measurement
    return sum([coords_to_rad_vspace(
        vorigin_head, vorigin_tail,
        get_pose_of(robot, behavior[idx_cur], robot.body.core_v2),
        get_pose_of(robot, behavior[idx_next], pmap["tail"]["box"])) 
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
        robot, behavior[0], pmap["tail"]["box"] 
    )
    
    n = len(behavior)
    state_pairs = list(zip(range(0, n, 2), range(1, n, 2)))
    
    # For all measurements, we add the total radians rotated from
    # the previous measurement
    rad_deltas = [
        coords_to_rad_vspace(
            vorigin_head, vorigin_tail,
            get_pose_of(robot, behavior[idx_cur], robot.body.core_v2),
            get_pose_of(robot, behavior[idx_next], pmap["tail"]["box"])
        ) 
     for idx_cur, idx_next in state_pairs]
    return sum(rad[0] for rad in rad_deltas if (go_left and rad[1] == "left") or (not go_left and rad[1] == "right"))

def get_pose_actualize_angle(robot: ModularRobot, behavior: simulated_behavior, p: float):
    """
    Use a gaussian distribution to reinforce using angles in the body instead of
    keeping the body straight. This corresponds mainly for the arms and legs.
    """
    body = PhysMap.map_with(robot.body)
    whitelist = [
        body["left_arm"]["hinge"],
        body["right_arm"]["hinge"],
        body["left_leg"]["hinge"],
        body["right_leg"]["hinge"],
    ]

    def eval_angle(s0: ModularRobotSimulationState):
        return sum(
            [norm.pdf(e,p,0.1) for e in [
                s0.get_hinge_position(h) for h in whitelist]]
        )

    total_states = len(behavior)

    # Return the sum of all probabilties that the hinge is at `p`
    return sum(
        [eval_angle(states.get_modular_robot_simulation_state(robot)) 
        for states in behavior]
    )

def evaluate_angle_with_projection_with_z_avg(
    robots: list[ModularRobot], behaviors: list[simulated_behavior], 
    run_idx: int, gen_idx: int
) -> npt.NDArray[np.float_]:
    target = (target_dist_per_run[run_idx-1] , target_angle_per_run[run_idx-1])

    def evalf(robot: ModularRobot, states: simulated_behavior):
        # return get_pose_cumulative_maximal_rotation(robot, states)
        # print(f"{get_pose_maximal_rotation_filter_dir(robot, states, go_left=False)=} - {get_pose_muscle_effiency(robot, states)=}")
        return get_pose_maximal_rotation_filter_dir(robot, states, go_left=True) - get_pose_muscle_effiency(robot, states)
    return np.array([
        evalf(robot, states) for robot, states in zip(robots, behaviors)
    ])

def evaluate_via_target_approx(robots: list[ModularRobot], behaviors: list[simulated_behavior], theta: float) -> npt.NDArray[np.float_]:
    """
    Perform an evaluation along a Polar 2D plane. Give an angle theta in radians
    and the robot will be measured against its projected distance among 
    that angle. 

    The procedure is to normalize the vector but retain its angle against the
    origin. Given angle theta, construct coordinate (1, theta). We measure the
    projected distance which will between values [0,1] where 0 is 0 is no 
    progress along the angle and 1 is complete progress.  
    """

    # In theory, we should make it such that the center is the robots head but 
    # since this system is trained with always the robot vaguely around the 
    # origin 0 this shouldnt be a big deal
    cart_target_vec = polar_to_cartesian_2d(5, theta)

    def evalf(robot: ModularRobot, states: simulated_behavior):
        optimize = get_pose_nd_projected_mag(robot, states, cart_target_vec)

        penalize = get_pose_actualize_angle(robot, states, 1)
        # penalize = get_pose_muscle_effiency(robot, states)
        print(f"{optimize=}\n{penalize=}")
        return optimize * penalize
        # return optimize - penalize / 2

    return np.array([
        evalf(robot, states) for robot,states in zip(robots, behaviors)
    ]) 

def evaluate_angle_actualization(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
    """
    Perform evaluation on how well the robot is able to retain an angle on its 
    limbs
    """
    return np.array([
        get_pose_actualize_angle(robot, behavior, 1)
        for robot, behavior in zip(robots, behaviors) 
    ])

def evaluate_pose_x_delta(robots: list[ModularRobot], behaviors: list[simulated_behavior]) -> npt.NDArray[np.float_]:
    """
    Perform evaluation over a list of robots. The incoming data is **assumed**
    to be ordered. I.E. the first index in the modular robot list has its 
    behavior recorded in the first index of the behavior list.
    
    Returns an array of ordered fitness values.
    """

    def evalf(robot: ModularRobot, states: simulated_behavior):
        optimize = get_pose_x_delta(
            states[0].get_modular_robot_simulation_state(robot),
            states[-1].get_modular_robot_simulation_state(robot)
        ) * 200
        penalize = get_pose_muscle_effiency(robot, states)
        print(f"{optimize=}\n{penalize=}")
        return optimize - penalize

    return np.array([
        evalf(robot, states) for robot, states in zip(robots, behaviors)
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
