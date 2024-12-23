from VAE import VAE_similarity,infer_on_csv
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import cv2
import os
import config











# In order to standardize animal and robot dataset, tuple should be split to X and Y.
# After this, they have same format
def parse_and_split_coordinates(df, coordinate_columns):
    for col in coordinate_columns:
        df[col] = df[col].apply(lambda coord: ast.literal_eval(coord) if isinstance(coord, str) else coord)
        # split the tuple into 2 values: x,y
        df[[f"{col}_x", f"{col}_y"]] = pd.DataFrame(df[col].tolist(), index=df.index)
    # delete original column
    df = df.drop(columns=coordinate_columns)
    return df


# MSE and DTW --> the less the better, we should inverse them
# Then, The higher, the better performance
#frame by frame
def mse_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    """
    逐帧计算 MSE 相似度并求平均值。
    """
    if robot_data.shape == animal_data.shape:
        mse_values = []
        for i in range(robot_data.shape[0]):  # 遍历每一帧
            mse = mean_squared_error(robot_data.iloc[i], animal_data.iloc[i])
            mse_values.append(mse)
        return np.mean(mse_values)  # 返回平均 MSE
    else:
        raise ValueError("Shape mismatch between robot and animal data, cannot calculate MSE.")


def dtw_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    """
    逐帧计算 DTW 相似度并求平均值。
    """
    if robot_data.shape == animal_data.shape:
        dtw_values = []
        for i in range(robot_data.shape[0]):  # 遍历每一帧
            distance, _ = fastdtw(robot_data.iloc[i], animal_data.iloc[i], dist=euclidean)
            dtw_values.append(distance)
        return np.mean(dtw_values)  # 返回平均 DTW
    else:
        raise ValueError("Shape mismatch between robot and animal data, cannot calculate DTW.")


def cos_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    """
    逐帧计算余弦相似度并求平均值。
    """
    if robot_data.shape == animal_data.shape:
        cos_values = []
        for i in range(robot_data.shape[0]):  # 遍历每一帧
            # 计算当前帧的余弦相似度
            cosine_matrix = cosine_similarity(
                robot_data.iloc[i].values.reshape(1, -1),
                animal_data.iloc[i].values.reshape(1, -1)
            )
            cos_values.append(cosine_matrix[0, 0])  # 提取相似度值
        return np.mean(cos_values)  # 返回平均余弦相似度
    else:
        raise ValueError("Shape mismatch between robot and animal data, cannot calculate COSINE.")


##batch


##as a whole
# def mse_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
#
#     # Ensure data shapes match before calculating MSE
#     if robot_data.shape == animal_data.shape:
#         mse = mean_squared_error(robot_data, animal_data)
#         return mse
#     else:
#         raise ValueError("Shape mismatch between robot and animal data, cannot calculate MSE.")
#
#
# def dtw_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
#     # Ensure data shapes match before calculating DTW
#     if robot_data.shape == animal_data.shape:
#         distance, path = fastdtw(robot_data, animal_data, dist=euclidean)
#         return distance
#     else:
#         raise ValueError("Shape mismatch between robot and animal data, cannot calculate DTW.")
#
#
# def cos_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
#     # Ensure data shapes match before calculating cosine
#     if robot_data.shape == animal_data.shape:
#         # cosine matrix
#         cosine_matrix = cosine_similarity(robot_data, animal_data)
#         # Compute the average cosine similarity
#         avg_cosine_value = cosine_matrix.mean()
#         return avg_cosine_value
#     else:
#         raise ValueError("Shape mismatch between robot and animal data, cannot calculate COSINE.")


def calculate_similarity(scale_data: pd.DataFrame, state: config.EAState) -> pd.DataFrame:
    similarity_type = state.similarity_type

    coordinate_columns = ['middle', 'rear', 'left_hind', 'left_front', 'head', 'right_hind', 'right_front']

    # read animal dataset
    # animal_data = pd.read_csv('./src/model/animal_data_head_orgin_884.csv').reset_index(drop=True)
    # animal_data = pd.read_csv('./src/model/slow_with_linear_4.csv').reset_index(drop=True)
    # animal_data = pd.read_csv('./src/model/CubicSpline_interpolated_3.csv').reset_index(drop=True)
    animal_data = state.animal_data.copy()
    animal_data = parse_and_split_coordinates(animal_data, coordinate_columns)

    similarities = []

    animal_data = animal_data.drop(columns=['Frame'])
    for robot_index in scale_data['robot_index'].unique():

        robot_data = scale_data[scale_data['robot_index'] == robot_index].drop(columns=['robot_index'])
        # make sure the robot frame is consistent with animal
        robot_data['Frame'] = range(len(robot_data))
        robot_data = parse_and_split_coordinates(robot_data, coordinate_columns)
        # print(f"Robots {robot_index}", robot_data)
        robot_data = robot_data.drop(columns=['Frame'])

        min_rows = min(len(robot_data), len(animal_data))
        robot_data = robot_data[:min_rows]
        animal_data = animal_data[:min_rows]
        # print(robot_data.head(2))
        # print(animal_data.head(5))

        try:
            if similarity_type == "MSE":
                mse = mse_similarity(robot_data, animal_data)
                logging.debug(f"MSE between Robot {robot_index} and animal: {mse}")
                similarities.append(mse)
            elif similarity_type == "DTW":
                dtw = dtw_similarity(robot_data, animal_data)
                logging.debug(f"DTW distance between Robot {robot_index} and animal: {dtw}")
                similarities.append(dtw)
            elif similarity_type == "Cosine":
                cos = cos_similarity(robot_data, animal_data)
                logging.debug(f"Cosine distance between Robot {robot_index} and animal: {cos}")
                similarities.append(cos)
            else:
                mse = mse_similarity(robot_data, animal_data)
                dtw = dtw_similarity(robot_data, animal_data)
                cos = cos_similarity(robot_data, animal_data)

                similarities.append([mse, dtw, cos])
        except ValueError as e:
            logging.error(
                f"Skipping animal for Robot {robot_index} due to shape mismatch.")

    return np.array(similarities)

def combination_fitnesses(distance, df_robot, state: config.EAState):
    df_animal = state.animal_data
    a = state.alpha
    similarity_type = state.similarity_type
    
    if  similarity_type == "DTW":
        animal_similarity = calculate_similarity(df_robot, state)
        animal_similarity=-animal_similarity
        animal_similarity=DTW_fitness_scaling(animal_similarity)

        distance = distance_fitness_scaling(distance)
        combination = -a * np.array(distance) - (1 - a) * np.array(animal_similarity)
        logging.debug("combined_fitnesses", combination)
        return combination, distance, animal_similarity
    elif similarity_type=="Cosine":
    # Cosine_similarity
        animal_similarity = calculate_similarity(df_robot,state)
        animal_similarity=Cosine_fitness_scaling(animal_similarity)
        distance=distance_fitness_scaling(distance)
        combination=-a*np.array(distance)-(1-a)*np.array(animal_similarity)
        logging.debug("combined_fitnesses", combination)
        return combination, distance, animal_similarity
    #MSE
    elif similarity_type=="MSE":
        animal_similarity = calculate_similarity(df_robot,state)
        animal_similarity=-animal_similarity
        animal_similarity=MSE_fitness_scaling(animal_similarity)
        distance=distance_fitness_scaling(distance)
        combination=-a*np.array(distance)-(1-a)*np.array(animal_similarity)
        logging.debug("combined_fitnesses",combination)
        return combination,distance,animal_similarity
    #VAE
    elif similarity_type=="VAE":
        df_robot = infer_on_csv(df_robot)
        animal_similarity=VAE_similarity(df_robot,df_animal)
        animal_similarity = (-1) * np.array(animal_similarity)
        animal_similarity=VAE_fitness_scaling(animal_similarity)
        distance=distance_fitness_scaling(distance)
        combination=-a*np.array(distance)-(1-a)*np.array(animal_similarity)
        logging.debug("combined_fitnesses",combination)
        return combination,distance,animal_similarity
    #
    else:
        # VAE
        VAE_robot = infer_on_csv(df_robot)
        vae_similarity = VAE_similarity(VAE_robot, df_animal)
        vae_similarity = (-1) * np.array(vae_similarity)
        vae_similarity = VAE_fitness_scaling(vae_similarity)

        # 计算 MSE, DTW, Cosine 相似度并进行 scaling
        MSE_DTW_Cosine_similarity = calculate_similarity(df_robot, state)

        # 对每种相似度进行 scaling
        mse_similarity = MSE_fitness_scaling([-similarity[0] for similarity in MSE_DTW_Cosine_similarity])
        dtw_similarity = DTW_fitness_scaling([-similarity[1] for similarity in MSE_DTW_Cosine_similarity])
        cosine_similarity = Cosine_fitness_scaling([similarity[2] for similarity in MSE_DTW_Cosine_similarity])

        # 合并所有相似度值
        combined_similarity = [
            list(mse_dtw_cos) + [vae] for mse_dtw_cos, vae in zip(
                zip(mse_similarity, dtw_similarity, cosine_similarity), vae_similarity
            )
        ]

        # 计算距离并进行 scaling
        distance = distance_fitness_scaling(distance)

        combination = (-1) * np.array(distance)
        return combination,distance,combined_similarity



def distance_fitness_scaling(fitnesses):
    # best 1 worst 0
    scaled_fitness=[]
    min_f=-0.2
    max_f=3
    # if min_f>np.min(fitnesses):
    #     min_f = np.min(fitnesses)
    if max_f<np.max(fitnesses):
        max_f = np.max(fitnesses)
    for i in fitnesses:
        if i < min_f:
            scaled_fitness.append(0)
        else:
            scaled_fitness.append((i - min_f) / (max_f - min_f))
    logging.debug('Distance_fitnesses',fitnesses)
    logging.debug('Distance_scaled_fitnesses',scaled_fitness)
    return scaled_fitness


def DTW_fitness_scaling(fitnesses):
    # best 1 worst 0
    scaled_fitness=[]
    min_f=-1100000
    max_f=-100000
    # if min_f > np.min(fitnesses):
    #     min_f = np.min(fitnesses)
    if max_f < np.max(fitnesses):
        max_f = np.max(fitnesses)
    for i in fitnesses:
        if i < min_f:
            scaled_fitness.append(0)
        else:
            scaled_fitness.append((i - min_f) / (max_f - min_f))
    logging.debug('DTW_fitnesses',fitnesses)
    logging.debug('DTW_scaled_fitnesses',scaled_fitness)
    return scaled_fitness

def Cosine_fitness_scaling(fitnesses):
    # best 0 worst 2
    fitnesses=-(1-np.array(fitnesses))
    scaled_fitness=[]
    min_f =-2
    max_f =0
    if min_f > np.min(fitnesses):
        min_f = np.min(fitnesses)
    if max_f < np.max(fitnesses):
        max_f = np.max(fitnesses)
    for i in fitnesses:
        scaled_fitness.append((i - min_f) / (max_f - min_f))
    logging.debug('Cosine_fitnesses',fitnesses)
    logging.debug('Cosine_scaled_fitnesses',scaled_fitness)
    return scaled_fitness


def VAE_fitness_scaling(fitnesses):
    # best 1 worst 0
    scaled_fitness=[]
    min_f =-18
    max_f =-15
    if min_f > np.min(fitnesses):
        min_f = np.min(fitnesses)
    if max_f < np.max(fitnesses):
        max_f = np.max(fitnesses)
    for i in fitnesses:
        scaled_fitness.append((i - min_f) / (max_f - min_f))
    logging.debug('VAE_fitnesses',fitnesses)
    logging.debug('VAE_scaled_fitnesses',scaled_fitness)
    return scaled_fitness


def MSE_fitness_scaling(fitnesses):
    # best 1 worst 0
    scaled_fitness=[]
    min_f=-210000
    max_f=-10000
    # if min_f > np.min(fitnesses):
    #     min_f = np.min(fitnesses)
    if max_f < np.max(fitnesses):
        max_f = np.max(fitnesses)
    for i in fitnesses:
        if i < min_f:
            scaled_fitness.append(0)
        else:
            scaled_fitness.append((i - min_f) / (max_f - min_f))
    logging.debug('MSE_fitnesses',fitnesses)
    logging.debug('MSE_scaled_fitnesses',scaled_fitness)
    return scaled_fitness







def create_simulation_video(state: config.EAState,frame_width=1280, frame_height=960, fps=10):
    torso=True
    run_id = state.max_runs
    alpha = state.alpha
    similarity_type = state.similarity_type

    data_robot_path=f"best-generations-{run_id}.csv"

    output_dir = f"results-{alpha}-{similarity_type}"
    # 定义输出视频路径
    output_video_path = os.path.join(output_dir, f'simulation_{run_id}-{alpha}-{similarity_type}.mp4')

    # Read animal and robot data
    data_animal = state.animal_data.copy()
    data_robot = pd.read_csv(data_robot_path)
    # visualize the last_generation
    last_generation_id = data_robot['generation_id'].max()
    data_robot=data_robot[data_robot['generation_id'] == last_generation_id]
    # print("\ndata_robot\n",data_robot)
    if torso == True:
        transformed_df = translate_and_rotate_torso(data_robot)
        scaled_robot_df = scale_robot_coordinates_torso(data_animal, transformed_df)
        scaled_robot_df = stationary_robot_movement_torso(scaled_robot_df)
        data_robot=scaled_robot_df
        data_animal = stationary_animal_movement_torso(data_animal)
    else:
        transformed_df = translate_and_rotate(data_robot)
        scaled_robot_df = scale_robot_coordinates(data_animal, transformed_df)
        scaled_robot_df = stationary_robot_movement(scaled_robot_df)
        data_robot=scaled_robot_df
        data_animal=stationary_animal_movement(data_animal)
    # black for animal red for robot
    robot_color = (0, 0, 255)
    animal_color = (0, 0, 0)
    # remove animal distance


    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    gif_frames = []

    center_x = frame_width // 2
    center_y = frame_height // 2

    min_length = min(len(data_robot), len(data_animal))
    data_robot = data_robot.iloc[:min_length]
    data_animal = data_animal.iloc[:min_length]

    for i in range(len(data_animal)):
        # Get the i-th row of robot and animal data
        frame_data_robot = data_robot.iloc[i]
        # print("\nframe_data_robot",frame_data_robot)
        frame_data_animal = data_animal.iloc[i]

        # Create white background
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        robot_points = {}
        animal_points = {}
        columns=['head', 'middle', 'rear', 'left_front', 'right_front', 'left_hind', 'right_hind']
        for column in columns:
            if column != 'Frame':
                if pd.notna(frame_data_robot[column]):
                    try:
                        coord = parse_tuple(frame_data_robot[column])
                        # print("\n coord",coord)
                        # print(frame_data_robot[column])
                        x = int(center_x + coord[0])
                        y = int(center_y - coord[1])
                        robot_points[column] = (x, y)  # Save the coordinates for connecting lines
                        cv2.circle(img, (x, y), 3, robot_color, -1)  # Robot points
                    except:
                        logging.error(f"Error parsing coordinates for {column} in row {i} of robot data")

        robot_connections = [('head', 'left_front'), ('head', 'middle'),('middle','rear'),('head', 'right_front'), ('rear', 'right_hind'), ('rear', 'left_hind')]  # Define connections
        for point1, point2 in robot_connections:
            if point1 in robot_points and point2 in robot_points:
                cv2.line(img, robot_points[point1], robot_points[point2], robot_color, 2)  # Line thickness = 2

        # Draw animal data and save points
        for column in frame_data_animal.index:
            if column != 'Frame':
                if pd.notna(frame_data_animal[column]) and isinstance(frame_data_animal[column], str):
                    try:
                        coord = eval(frame_data_animal[column])

                        x = int(center_x + coord[0])
                        y = int(center_y - coord[1])
                        animal_points[column] = (x, y)
                        cv2.circle(img, (x, y), 3, animal_color, -1)  # Animal points
                    except:
                        logging.error(f"Error parsing coordinates for {column} in row {i} of animal data")

        # Connect animal points with lines
        animal_connections = [('head', 'left_front'),('head', 'middle'),('middle','rear'), ('head', 'right_front'), ('rear', 'right_hind'), ('rear', 'left_hind')]  # Define connections
        for point1, point2 in animal_connections:
            if point1 in animal_points and point2 in animal_points:
                cv2.line(img, animal_points[point1], animal_points[point2], animal_color, 2)  # Line thickness = 2

        # Add frame to video
        output_video.write(img)

        # Save frame for GIF (if needed)
        gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Finalize video
    output_video.release()
    cv2.destroyAllWindows()

    return gif_frames

def parse_tuple(value):
    try:
        # 尝试将字符串解析为数值元组
        return np.array(ast.literal_eval(value)) if isinstance(value, str) else np.array(value)
    except (ValueError, SyntaxError):
        logging.error(f"Error parsing value: {value}")
        return np.array([np.nan, np.nan])


def calculate_and_add_center(df):
    """
    计算中心点 (center-euclidian) 并添加到数据帧中。

    Args:
        df (pd.DataFrame): 包含关键点数据的输入数据帧。

    Returns:
        pd.DataFrame: 添加了 center-euclidian 列的新数据帧。
    """
    # 定义需要的关键点列
    keypoints = ['head', 'middle', 'rear', 'right_front', 'left_front', 'right_hind', 'left_hind']

    # 确保数据帧中包含所有关键点列ls

    missing_columns = [col for col in keypoints if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    def calculate_center(row):
        """
        计算当前行的中心点坐标。
        """
        points = []
        for key in keypoints:
            try:
                point = parse_tuple(row[key])
                if isinstance(point, (tuple, list, np.ndarray)) and not np.isnan(point).any():
                    points.append(np.array(point))
            except Exception as e:
                logging.error(f"Error parsing {key}: {e}")

        # 如果找到有效的关键点，计算它们的平均值
        if points:
            center = np.mean(points, axis=0)
            return f"({center[0]:.2f}, {center[1]:.2f})"
        else:
            return np.nan

    # 应用到每一行，计算中心点
    df['center-euclidian'] = df.apply(calculate_center, axis=1)
    return df


def translate_and_rotate(df):
    df = calculate_and_add_center(df)
    required_columns = ['head']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {', '.join(missing_columns)}. Skipping file.")
        return None  # Skip if any required columns are missing

    # Parse 'center-euclidian' and 'head' as tuples
    center_str = parse_tuple(df['center-euclidian'].iloc[0])
    head_str = parse_tuple(df['head'].iloc[0])
    # 检查中心点和头部点是否有效
    if not isinstance(center_str, (tuple, list, np.ndarray)) or not isinstance(head_str, (tuple, list, np.ndarray)):
        print(f"Invalid format for center or head in first frame: center={center_str}, head={head_str}")
        return None
    # Convert head and center points to np.array if they are not
    center_str = np.array(center_str)
    head_str = np.array(head_str)
    # 计算 forward 列
    df['forward'] = df['head'].apply(
        lambda h: parse_tuple(h) - center_str if isinstance(h, str) else np.array([np.nan, np.nan]))
    forward_str = df['forward'].iloc[0]

    try:
        head_x, head_y = head_str
        center_x, center_y = center_str
        forward_x, forward_y = forward_str
    except ValueError:
        print(f"Error parsing center or forward in the first frame: {center_str}, {forward_str}")
        return None

    theta = (np.pi / 2) - np.arctan2(forward_y, forward_x)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    transformed_data = []

    for index, row in df.iterrows():
        transformed_points = {}

        transformed_points['generation_best_fitness_score'] = row.get('generation_best_fitness_score', np.nan)
        transformed_points['generation_id'] = row.get('generation_id', np.nan)
        transformed_points['frame_id'] = row.get('frame_id', np.nan)
        points_to_transform = {
            'head': parse_tuple(row['head']),
            'center-euclidian': parse_tuple(row['center-euclidian']),
            'forward': parse_tuple(row['forward']),
            'middle': parse_tuple(row['middle']),
            'rear': parse_tuple(row['rear']),
            'right_front': parse_tuple(row['right_front']),
            'left_front': parse_tuple(row['left_front']),
            'right_hind': parse_tuple(row['right_hind']),
            'left_hind': parse_tuple(row['left_hind']),
        }
        for point_name, point_value in points_to_transform.items():
            try:
                if np.isnan(point_value).any():
                    print(f"Skipping {point_name} due to missing value.")
                    continue
                # center
                # translated_point = point_value - head_str
                translated_point = point_value - center_str
                rotated_point = rotation_matrix @ translated_point
                transformed_points[point_name] = f"({rotated_point[0]:.2f}, {rotated_point[1]:.2f})"

            except Exception as e:
                print(f"Error parsing {point_name} with value {point_value}: {e}")

        transformed_data.append(transformed_points)

    return pd.DataFrame(transformed_data)


def translate_and_rotate_torso(df):
    df = calculate_and_add_center(df)
    required_columns = ['head','middle']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {', '.join(missing_columns)}. Skipping file.")
        return None  # Skip if any required columns are missing

    # Parse 'center-euclidian' and 'head' as tuples
    center_str = parse_tuple(df['center-euclidian'].iloc[0])
    head_str = parse_tuple(df['head'].iloc[0])
    middle_str = parse_tuple(df['middle'].iloc[0])
    # 检查中心点和头部点是否有效
    if not isinstance(center_str, (tuple, list, np.ndarray)) or not isinstance(head_str, (tuple, list, np.ndarray)):
        print(f"Invalid format for center or head in first frame: center={center_str}, head={head_str}")
        return None
    # Convert head and center points to np.array if they are not
    center_str = np.array(center_str)
    head_str = np.array(head_str)
    middle_str = np.array(middle_str)
    # 计算 forward 列
    df['forward'] = df['head'].apply(
        lambda h: parse_tuple(h) - center_str if isinstance(h, str) else np.array([np.nan, np.nan]))
    forward_str = df['forward'].iloc[0]

    try:
        head_x, head_y = head_str
        center_x, center_y = center_str
        forward_x, forward_y = forward_str
    except ValueError:
        print(f"Error parsing center or forward in the first frame: {center_str}, {forward_str}")
        return None

    theta = (np.pi / 2) - np.arctan2(forward_y, forward_x)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    transformed_data = []

    for index, row in df.iterrows():
        transformed_points = {}

        transformed_points['generation_best_fitness_score'] = row.get('generation_best_fitness_score', np.nan)
        transformed_points['generation_id'] = row.get('generation_id', np.nan)
        transformed_points['frame_id'] = row.get('frame_id', np.nan)
        points_to_transform = {
            'head': parse_tuple(row['head']),
            'center-euclidian': parse_tuple(row['center-euclidian']),
            'forward': parse_tuple(row['forward']),
            'middle': parse_tuple(row['middle']),
            'rear': parse_tuple(row['rear']),
            'right_front': parse_tuple(row['right_front']),
            'left_front': parse_tuple(row['left_front']),
            'right_hind': parse_tuple(row['right_hind']),
            'left_hind': parse_tuple(row['left_hind']),
        }
        for point_name, point_value in points_to_transform.items():
            try:
                if np.isnan(point_value).any():
                    print(f"Skipping {point_name} due to missing value.")
                    continue
                # head
                # translated_point = point_value - head_str
                #center
                # translated_point = point_value - center_str
                #torso
                translated_point = point_value - middle_str
                rotated_point = rotation_matrix @ translated_point
                transformed_points[point_name] = f"({rotated_point[0]:.2f}, {rotated_point[1]:.2f})"

            except Exception as e:
                print(f"Error parsing {point_name} with value {point_value}: {e}")

        transformed_data.append(transformed_points)

    return pd.DataFrame(transformed_data)



#  "(x, y)" ->(x, y)
def parse_tuple_string(s):
    if s:
        return tuple(map(float, s.strip('()').split(',')))
    return None




def scale_robot_coordinates(df_animal, df_robot):
    df_robot = calculate_and_add_center(df_robot)
    df_animal = calculate_and_add_center(df_animal)
    coordinates_2_list = []
    for index, frame in df_robot.iterrows():
        coordinates_2 = {
            'center':parse_tuple(frame['center-euclidian']),
            'head': parse_tuple_string(frame['head']),
            'middle': parse_tuple_string(frame['middle']),
            'rear': parse_tuple_string(frame['rear']),
            'right_front': parse_tuple_string(frame['right_front']),
            'left_front': parse_tuple_string(frame['left_front']),
            'right_hind': parse_tuple_string(frame['right_hind']),
            'left_hind': parse_tuple_string(frame['left_hind']),
        }
        coordinates_2_list.append(coordinates_2)

    global_max_distance_robot = 0
    global_max_distance_animal = 0

    max_frames = min(len(coordinates_2_list), len(df_animal))
    df_animal = df_animal.iloc[:max_frames]
    coordinates_2_list = coordinates_2_list[:max_frames]
    for i, frame in df_animal.iterrows():
        animal_center = parse_tuple_string(frame['center-euclidian'])
        if animal_center is None:
            print(f"Frame {i}: Missing animal head coordinates.")
            continue


        animal_coordinates = [
            parse_tuple_string(frame[col]) for col in
            ['middle', 'rear', 'head','left_front', 'left_hind', 'right_hind', 'right_front']
        ]
        if None in animal_coordinates:
            print(f"Frame {i}: Missing some animal keypoints, skipping.")
            continue

        distances = np.linalg.norm(np.array(animal_coordinates) - np.array(animal_center), axis=1)
        max_distance_animal = np.max(distances)

        if max_distance_animal > global_max_distance_animal:
            global_max_distance_animal = max_distance_animal
        # 机器人最大距离计算
        robot_coordinates = list(coordinates_2_list[i].values())
        robot_center = np.array(coordinates_2_list[i]['center'])
        robot_distances = np.linalg.norm(np.array(robot_coordinates) - robot_center, axis=1)
        max_distance_robot = np.max(robot_distances)

        if max_distance_robot > global_max_distance_robot:
            global_max_distance_robot = max_distance_robot

    scaling_factor = global_max_distance_animal / global_max_distance_robot
    print(f"Scaling factor: {scaling_factor}")
    # 缩放机器人的数据
    scaled_robot_data = []
    first_robot_center = np.array(coordinates_2_list[0]['center'])
    print("first_robot_center",first_robot_center)
    for i in range(len(coordinates_2_list)):
        robot_coordinates = list(coordinates_2_list[i].values())
        scaled_robot_coordinates = (np.array(robot_coordinates) - first_robot_center) * scaling_factor + first_robot_center

        scaled_robot_data.append({
            'Frame': i,
            'center':tuple(scaled_robot_coordinates[0]),
            'head': tuple(scaled_robot_coordinates[1]),
            'middle': tuple(scaled_robot_coordinates[2]),
            'rear': tuple(scaled_robot_coordinates[3]),
            'right_front': tuple(scaled_robot_coordinates[4]),
            'left_front': tuple(scaled_robot_coordinates[5]),
            'right_hind': tuple(scaled_robot_coordinates[6]),
            'left_hind': tuple(scaled_robot_coordinates[7]),
        })
    scaled_robot_data=pd.DataFrame(scaled_robot_data)
    # print("\nscaled_robot_data\n",scaled_robot_data.head(15))
    return scaled_robot_data


def scale_robot_coordinates_torso(df_animal, df_robot):

    df_robot = calculate_and_add_center(df_robot)
    df_animal = calculate_and_add_center(df_animal)
    coordinates_2_list = []
    for index, frame in df_robot.iterrows():
        coordinates_2 = {
            'center':parse_tuple(frame['center-euclidian']),
            'head': parse_tuple_string(frame['head']),
            'middle': parse_tuple_string(frame['middle']),
            'rear': parse_tuple_string(frame['rear']),
            'right_front': parse_tuple_string(frame['right_front']),
            'left_front': parse_tuple_string(frame['left_front']),
            'right_hind': parse_tuple_string(frame['right_hind']),
            'left_hind': parse_tuple_string(frame['left_hind']),
        }
        coordinates_2_list.append(coordinates_2)

    # global_max_distance_robot = 0
    # global_max_distance_animal = 0

    max_frames = min(len(coordinates_2_list), len(df_animal))
    df_animal = df_animal.iloc[:max_frames]
    coordinates_2_list = coordinates_2_list[:max_frames]

    # 提取动物关键点
    for i, frame in df_animal.iterrows():
        animal_middle= parse_tuple_string(frame['middle'])
        if animal_middle is None:
            print(f"Frame {i}: Missing animal middle coordinates.")
            continue

        # 提取所有关键点并计算到 center 的距离
        animal_coordinates = [
            parse_tuple_string(frame[col]) for col in
            ['middle', 'rear', 'head','left_front', 'left_hind', 'right_hind', 'right_front']
        ]
        if None in animal_coordinates:
            print(f"Frame {i}: Missing some animal keypoints, skipping.")
            continue

        # distances = np.linalg.norm(np.array(animal_coordinates) - np.array(animal_middle), axis=1)
        # max_distance_animal = np.max(distances)
        #
        # if max_distance_animal > global_max_distance_animal:
        #     global_max_distance_animal = max_distance_animal
        # # 机器人最大距离计算
        # robot_coordinates = list(coordinates_2_list[i].values())
        # robot_middle = np.array(coordinates_2_list[i]['middle'])
        # robot_distances = np.linalg.norm(np.array(robot_coordinates) - robot_middle, axis=1)
        # max_distance_robot = np.max(robot_distances)
        #
        # if max_distance_robot > global_max_distance_robot:
        #     global_max_distance_robot = max_distance_robot
    #
    # scaling_factor = global_max_distance_animal / global_max_distance_robot
    # print(f"Scaling factor: {scaling_factor}")
    scaling_factor=469.24713993617365

    # 缩放机器人的数据
    scaled_robot_data = []
    first_robot_middle = np.array(coordinates_2_list[0]['middle'])
    print("first_robot_middle",first_robot_middle)
    for i in range(len(coordinates_2_list)):
        robot_coordinates = list(coordinates_2_list[i].values())
        scaled_robot_coordinates = (np.array(robot_coordinates) - first_robot_middle) * scaling_factor + first_robot_middle

        scaled_robot_data.append({
            'Frame': i,
            'center':tuple(scaled_robot_coordinates[0]),
            'head': tuple(scaled_robot_coordinates[1]),
            'middle': tuple(scaled_robot_coordinates[2]),
            'rear': tuple(scaled_robot_coordinates[3]),
            'right_front': tuple(scaled_robot_coordinates[4]),
            'left_front': tuple(scaled_robot_coordinates[5]),
            'right_hind': tuple(scaled_robot_coordinates[6]),
            'left_hind': tuple(scaled_robot_coordinates[7]),
        })
    scaled_robot_data=pd.DataFrame(scaled_robot_data)
    # print("\nscaled_robot_data\n",scaled_robot_data.head(15))
    return scaled_robot_data








def extract_keypoints(data_dict):
    keypoints = []

    if isinstance(data_dict, dict):
        for name, coord in data_dict.items():
            if coord != (None, None):
                keypoints.append(coord)
    elif isinstance(data_dict, pd.DataFrame) or isinstance(data_dict, pd.Series):
        keypoints = data_dict[['X (relative)', 'Y (relative)']].values.flatten()

    return np.array(keypoints).flatten()  # 将二维坐标转为一维向量


def create_simulation_video_with_distance(run_id: int, alpha, similarity_type,df_animal , frame_width=1280, frame_height=960, fps=13, duration_seconds=27):
    data_animal_path = "./src/model/slow_with_linear_4.csv"

    #read the robot keypoints
    data_robot_path = f"results-{alpha}-{similarity_type}/raw_data-{run_id}.csv"
    data_best_path = f"results-{alpha}-{similarity_type}/last-generations-{run_id}-handled.csv"

    output_dir = f"results-{alpha}-{similarity_type}"
    output_video_path = os.path.join(output_dir, f'simulation_{run_id}-{alpha}-{similarity_type}_with_distance.mp4')

    # Read animal and robot data
    data_animal = pd.read_csv(data_animal_path)

    data_robot = pd.read_csv(data_robot_path)
    # visualize the last_generation
    last_generation_id = data_robot['generation_id'].max()
    data_robot = data_robot[data_robot['generation_id'] == last_generation_id]
    transformed_df = translate_and_rotate(data_robot)

    scaled_robot_df = scale_robot_coordinates(data_animal, transformed_df)
    # remove robot distance
    # scaled_robot_df=stationary_robot_movement(scaled_robot_df)
    # remove animal distance
    data_animal=stationary_animal_movement(data_animal)
    # save the last generation
    pd.DataFrame(scaled_robot_df).to_csv(data_best_path, index=False)
    #保存vae文件
    # save_vae_infer_csv(run_id,infer_on_csv(scaled_robot_df ),df_animal,alpha,similarity_type)

    data_robot = scaled_robot_df

    robot_color = (0, 0, 255)
    animal_color = (0 ,0, 0)

    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    gif_frames = []

    center_x = frame_width // 2
    center_y = frame_height // 2

    min_length = min(len(data_robot), len(data_animal))
    data_robot = data_robot.iloc[:min_length]
    data_animal = data_animal.iloc[:min_length]

    # 计算最多需要多少帧（20秒 * fps）
    max_frames = min(len(data_animal), duration_seconds * fps)

    for i in range(max_frames):
        # Get the i-th row of robot and animal data
        frame_data_robot = data_robot.iloc[i]
        frame_data_animal = data_animal.iloc[i]

        # Create white background
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        robot_points = {}
        animal_points = {}
        columns = ['head', 'middle', 'rear', 'left_front', 'right_front', 'left_hind', 'right_hind']
        for column in columns:
            if column != 'Frame':
                if pd.notna(frame_data_robot[column]):
                    try:
                        coord = parse_tuple(frame_data_robot[column])
                        x = int(center_x + coord[0])
                        y = int(center_y - coord[1])
                        robot_points[column] = (x, y)
                        cv2.circle(img, (x, y), 3, robot_color, -1)
                    except:
                        print(f"Error parsing coordinates for {column} in row {i} of robot data")

        robot_connections = [('head', 'left_front'), ('head', 'middle'), ('middle', 'rear'),
                             ('head', 'right_front'), ('rear', 'right_hind'), ('rear', 'left_hind')]
        for point1, point2 in robot_connections:
            if point1 in robot_points and point2 in robot_points:
                cv2.line(img, robot_points[point1], robot_points[point2], robot_color, 2)

        # Draw animal data and save points
        for column in frame_data_animal.index:
            if column != 'Frame':
                if pd.notna(frame_data_animal[column]) and isinstance(frame_data_animal[column], str):
                    try:
                        coord = eval(frame_data_animal[column])
                        x = int(center_x + coord[0])
                        y = int(center_y - coord[1])
                        animal_points[column] = (x, y)
                        cv2.circle(img, (x, y), 3, animal_color, -1)
                    except:
                        print(f"Error parsing coordinates for {column} in row {i} of animal data")

        animal_connections = [('head', 'left_front'), ('head', 'middle'), ('middle', 'rear'),
                              ('head', 'right_front'), ('rear', 'right_hind'), ('rear', 'left_hind')]
        for point1, point2 in animal_connections:
            if point1 in animal_points and point2 in animal_points:
                cv2.line(img, animal_points[point1], animal_points[point2], animal_color, 2)

        # Add frame to video
        output_video.write(img)

        # Save frame for GIF (if needed)
        gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Finalize video
    output_video.release()
    cv2.destroyAllWindows()

    return gif_frames


def stationary_animal_movement(data: pd.DataFrame) -> pd.DataFrame:
    """
    调整动物数据，让中心点固定在原点，其他关键点相对中心点移动，保留姿态变化。

    Args:
        data (pd.DataFrame): 包含关键点数据的 DataFrame，格式为字符串的 "(x, y)"。

    Returns:
        pd.DataFrame: 调整后的 DataFrame，每帧的关键点以中心点为基准调整位置。
    """
    # 提取关键点列（排除 'Frame'）
    keypoint_columns = data.columns[1:]

    # 创建一个新的 DataFrame
    stationary_data = data.copy()
    # 计算第一帧的中心点
    first_frame_keypoints = []
    for col in keypoint_columns:
        point = ast.literal_eval(data.iloc[0][col])  # 解析第一帧的关键点
        first_frame_keypoints.append(point)
    first_frame_center = pd.DataFrame(first_frame_keypoints, columns=['x', 'y']).mean(axis=0)  # 第一帧中心点

    # 遍历每一帧数据
    for index, row in data.iterrows():
        keypoints = []

        # 解析每个关键点的坐标
        for col in keypoint_columns:
            point = ast.literal_eval(row[col])  # 将字符串 "(x, y)" 转为元组 (x, y)
            keypoints.append(point)

        # 计算当前帧的中心点
        keypoints = pd.DataFrame(keypoints, columns=['x', 'y'])
        center = keypoints.mean(axis=0)  # 当前帧中心点 (x, y)

        # 调整关键点：以中心点为基准计算相对位置
        adjusted_keypoints = keypoints - center

        # 更新到新的 DataFrame 中
        for col, adjusted_point in zip(keypoint_columns, adjusted_keypoints.values):
            stationary_data.at[index, col] = f"({adjusted_point[0]:.2f}, {adjusted_point[1]:.2f})"

    return stationary_data



def stationary_robot_movement(data: pd.DataFrame) -> pd.DataFrame:
    """
    调整机器人数据，让中心点固定在原点，其他关键点相对中心点移动，保留姿态变化。

    Args:
        data (pd.DataFrame): 包含关键点数据的 DataFrame，可以是字符串格式的 "(x, y)" 或元组格式的 (x, y)。

    Returns:
        pd.DataFrame: 调整后的 DataFrame，每帧的关键点以中心点为基准调整位置。
    """
    keypoint_columns = ['middle', 'rear', 'head', 'right_front', 'left_front', 'right_hind', 'left_hind']

    # 创建一个新的 DataFrame
    stationary_data = data.copy()

    # 计算第一帧的中心点
    first_frame_keypoints = []
    for col in keypoint_columns:
        point = data.iloc[0][col]
        # 如果是字符串则解析为元组
        if isinstance(point, str):
            point = ast.literal_eval(point)
        first_frame_keypoints.append(point)
    # 计算第一帧中心点
    first_frame_center = pd.DataFrame(first_frame_keypoints, columns=['x', 'y']).mean(axis=0)

    # 遍历每一帧数据
    for index, row in data.iterrows():
        keypoints = []

        # 解析每个关键点的坐标
        for col in keypoint_columns:
            point = row[col]
            # 如果是字符串则解析为元组
            if isinstance(point, str):
                point = ast.literal_eval(point)
            keypoints.append(point)

        # 计算当前帧的中心点
        keypoints = pd.DataFrame(keypoints, columns=['x', 'y'])
        center = keypoints.mean(axis=0)  # 当前帧中心点 (x, y)

        # 调整关键点：以中心点为基准计算相对位置
        adjusted_keypoints = keypoints - center

        # 更新到新的 DataFrame 中
        for col, adjusted_point in zip(keypoint_columns, adjusted_keypoints.values):
            stationary_data.at[index, col] = f"({adjusted_point[0]:.2f}, {adjusted_point[1]:.2f})"

    return stationary_data


def stationary_robot_movement_torso(data: pd.DataFrame) -> pd.DataFrame:
    """
    调整机器人数据，让 torso (middle) 固定在原点，其他关键点相对 torso 移动，保留姿态变化。

    Args:
        data (pd.DataFrame): 包含关键点数据的 DataFrame，可以是字符串格式的 "(x, y)" 或元组格式的 (x, y)。

    Returns:
        pd.DataFrame: 调整后的 DataFrame，每帧的关键点以 torso 为基准调整位置。
    """
    keypoint_columns = ['middle', 'rear', 'head', 'right_front', 'left_front', 'right_hind', 'left_hind']

    # 创建一个新的 DataFrame
    stationary_data = data.copy()

    # 遍历每一帧数据
    for index, row in data.iterrows():
        # 提取 middle (torso) 坐标作为基准点
        torso = row['middle']
        # 如果是字符串则解析为元组
        if isinstance(torso, str):
            torso = ast.literal_eval(torso)

        # 检查 torso 是否有效
        if not isinstance(torso, (tuple, list, np.ndarray)) or len(torso) != 2:
            print(f"Frame {index}: Invalid torso point: {torso}, skipping this frame.")
            continue

        # 解析每个关键点并调整为相对 torso 的位置
        for col in keypoint_columns:
            point = row[col]
            # 如果是字符串则解析为元组
            if isinstance(point, str):
                point = ast.literal_eval(point)

            # 检查关键点是否有效
            if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
                print(f"Frame {index}, Keypoint {col}: Invalid point: {point}, skipping this keypoint.")
                stationary_data.at[index, col] = f"(nan, nan)"
                continue

            # 调整关键点位置
            adjusted_point = np.array(point) - np.array(torso)
            stationary_data.at[index, col] = f"({adjusted_point[0]:.2f}, {adjusted_point[1]:.2f})"

    return stationary_data


def stationary_animal_movement_torso(data: pd.DataFrame) -> pd.DataFrame:
    """
    调整动物数据，让 torso (middle) 固定在原点，其他关键点相对 torso 移动，保留姿态变化。

    Args:
        data (pd.DataFrame): 包含关键点数据的 DataFrame，格式为字符串的 "(x, y)"。

    Returns:
        pd.DataFrame: 调整后的 DataFrame，每帧的关键点以 torso 为基准调整位置。
    """
    # 提取关键点列（排除 'Frame' 或其他非关键点列）
    keypoint_columns = [col for col in data.columns if col != 'Frame']

    # 创建一个新的 DataFrame
    stationary_data = data.copy()

    # 遍历每一帧数据
    for index, row in data.iterrows():
        # 提取 torso (middle) 坐标作为基准点
        torso = row['middle']
        # 如果是字符串则解析为元组
        if isinstance(torso, str):
            torso = ast.literal_eval(torso)

        # 检查 torso 是否有效
        if not isinstance(torso, (tuple, list, np.ndarray)) or len(torso) != 2:
            print(f"Frame {index}: Invalid torso point: {torso}, skipping this frame.")
            continue

        # 解析每个关键点并调整为相对 torso 的位置
        for col in keypoint_columns:
            point = row[col]
            # 如果是字符串则解析为元组
            if isinstance(point, str):
                point = ast.literal_eval(point)

            # 检查关键点是否有效
            if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
                print(f"Frame {index}, Keypoint {col}: Invalid point: {point}, skipping this keypoint.")
                stationary_data.at[index, col] = f"(nan, nan)"
                continue

            # 调整关键点位置
            adjusted_point = np.array(point) - np.array(torso)
            stationary_data.at[index, col] = f"({adjusted_point[0]:.2f}, {adjusted_point[1]:.2f})"

    return stationary_data