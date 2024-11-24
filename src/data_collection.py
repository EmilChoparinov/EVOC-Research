from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic

import pandas as pd
import numpy as np
import config
import os

from config import body_to_csv_map
from typedef import simulated_behavior

def record_cpg(robot: ModularRobot, run_id: int):
    filename = config.best_solution_per_ea
    brain: BrainCpgNetworkStatic = robot.brain
    pd.DataFrame(np.array(brain._weight_matrix)).to_csv(f"best-cpg-gen-{run_id}.csv", index=False, header=False)
    # return NotImplementedError()

def record_behavior(robot: ModularRobot, fitness: float, behavior: simulated_behavior, generation_id: int = -1, alpha: float = 1.0, fitness_function: str = "distance"):
# def record_behavior(robot: ModularRobot, fitness: float, behavior: simulated_behavior, generation_id: int = -1):
    # HACK: Type here is body but expects BodyV2. The type match is guarenteed
    #       because we use `gecko_v2` to construct the body, which uses the
    #       BodyV2 subtype. 
    csv_map = config.body_to_csv_map(robot.body) 


    for idx, state in enumerate(behavior):
        pose_func = state.get_modular_robot_simulation_state(robot).get_module_absolute_pose
        robot_coord_list = []
        
        def col_map(col: str):
            match col:
                case "generation_id": return generation_id
                case "center-euclidian": return 0 # calculate this after
                case "generation_best_fitness_score": return fitness
                case "frame_id": return idx
                case "alpha": return alpha
                case "fitness_function": return fitness_function
                case _:
                    abs_pose = pose_func(csv_map[col])
                    robot_coord_list.append((abs_pose.position.x, abs_pose.position.y))
                    return f"({abs_pose.position.x},{abs_pose.position.y})"
        
        # Collect the robots coordinates and put it in a dictionary that matches
        # the CSV definition `csv_cols`
        row = {col: col_map(col) for col in config.csv_cols}
        
        pd_coord_list = pd.DataFrame(robot_coord_list, columns=['x', 'y'])
        row["center-euclidian"] = f"({pd_coord_list['x'].mean()},{pd_coord_list['y'].mean()})"
        
        config.write_buffer.loc[len(config.write_buffer.index)] = row

def record_elite_generations(run_id: int, generation: int, fitness: float, matrix, alpha: float = 1.0, fitness_function: str = "distance"):
    # 2D to 1D
    flat_weights = np.array(matrix).flatten()
    data = {
        "Generation": [generation],
        "Fitness": [fitness],
        "Alpha": [alpha],
        "Fitness_Function": [fitness_function],
        **{f"Param_{i}": [weight] for i, weight in enumerate(flat_weights)}
    }
    df = pd.DataFrame(data)

    filename = f"elite-generations-{run_id}.csv"
    header = not os.path.isfile(filename)
    df.to_csv(filename, mode='a', index=False, header=not os.path.isfile(filename)) #pd.io.common.file_exists(filename))


# def record_best_fitness_generation_csv(
#         robot: ModularRobot,
#         fitness: float,
#         behavior: simulated_behavior,
#         generation_id: int,
#         output_file: str = "best_generations.csv",
#         alpha: float = 1.0,
#         fitness_function: str = "distance"
# ):
#     """
#     记录每一代的最佳个体到 CSV 文件，包括行为坐标。
#
#     Args:
#         generation_id (int): 当前代数。
#         robot (ModularRobot): 最优机器人对象。
#         fitness (float): 最优适应度值。
#         behavior (simulated_behavior): 最优行为数据。
#         output_file (str): 输出的 CSV 文件名。
#         alpha (float): 混合参数 alpha。
#         fitness_function (str): 使用的适应度函数类型。
#     """
#     if not os.path.exists(output_file):
#         with open(output_file, 'w') as f:
#             f.write("generation_id,fitness,alpha,fitness_function,head,middle,rear,right_front,left_front,right_hind,left_hind\n")
#     csv_map = config.body_to_csv_map(robot.body)
#     for idx, state in enumerate(behavior):
#         try:
#             pose_func = state.get_modular_robot_simulation_state(robot).get_module_absolute_pose
#             module_coords = {
#                 module_name: pose_func(csv_map[module_name]).position
#                 for module_name in ["head", "middle", "rear", "right_front", "left_front", "right_hind", "left_hind"]
#             }
#             # print(module_coords)
#             record = {
#                 "generation_id": generation_id,
#                 "fitness": fitness,
#                 "alpha": alpha,
#                 "fitness_function": fitness_function,
#                 "head": f"({module_coords['head'].x:.2f},{module_coords['head'].y:.2f})",
#                 "middle": f"({module_coords['middle'].x:.2f},{module_coords['middle'].y:.2f})",
#                 "rear": f"({module_coords['rear'].x:.2f},{module_coords['rear'].y:.2f})",
#                 "right_front": f"({module_coords['right_front'].x:.2f},{module_coords['right_front'].y:.2f})",
#                 "left_front": f"({module_coords['left_front'].x:.2f},{module_coords['left_front'].y:.2f})",
#                 "right_hind": f"({module_coords['right_hind'].x:.2f},{module_coords['right_hind'].y:.2f})",
#                 "left_hind": f"({module_coords['left_hind'].x:.2f},{module_coords['left_hind'].y:.2f})"
#             }
#             df = pd.DataFrame([record])
#             df.to_csv(output_file, mode='a', index=False, header=False)
#             # print(f"Frame {idx} of generation {generation_id} saved to {output_file}")
#         except Exception as e:
#             print(f"Error processing frame {idx} in generation {generation_id}: {e}")



