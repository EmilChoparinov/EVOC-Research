from VAE import VAE_similarity,infer_on_csv
import numpy as np
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast


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
def mse_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:

    # Ensure data shapes match before calculating MSE
    if robot_data.shape == animal_data.shape:
        mse = mean_squared_error(robot_data, animal_data)
        return mse
    else:
        raise ValueError("Shape mismatch between robot and animal data, cannot calculate MSE.")


def dtw_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    # Ensure data shapes match before calculating DTW
    if robot_data.shape == animal_data.shape:
        distance, path = fastdtw(robot_data, animal_data, dist=euclidean)
        return distance
    else:
        raise ValueError("Shape mismatch between robot and animal data, cannot calculate DTW.")


def cos_similarity(robot_data: pd.DataFrame, animal_data: pd.DataFrame) -> float:
    # Ensure data shapes match before calculating cosine
    if robot_data.shape == animal_data.shape:
        # cosine matrix
        cosine_matrix = cosine_similarity(robot_data, animal_data)
        # Compute the average cosine similarity
        avg_cosine_value = cosine_matrix.mean()
        return avg_cosine_value
    else:
        raise ValueError("Shape mismatch between robot and animal data, cannot calculate COSINE.")

def calculate_similarity(scale_data: pd.DataFrame,similarity_type) -> pd.DataFrame:

    coordinate_columns = ['middle', 'rear', 'left_hind', 'left_front', 'head', 'right_hind', 'right_front']

    # read animal dataset
    # animal_data = pd.read_csv('./src/model/animal_data_head_orgin_884.csv').reset_index(drop=True)
    animal_data = pd.read_csv('./src/model/slow_interpolated_4.csv').reset_index(drop=True)
    # animal_data = pd.read_csv('./src/model/CubicSpline_interpolated_3.csv').reset_index(drop=True)
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
                print(f"MSE between Robot {robot_index} and animal: {mse}")
                similarities.append(mse)
            elif similarity_type == "DTW":
                dtw = dtw_similarity(robot_data, animal_data)
                print(f"DTW distance between Robot {robot_index} and animal: {dtw}")
                similarities.append(dtw)
            elif similarity_type == "Cosine":
                cos = cos_similarity(robot_data, animal_data)
                print(f"Cosine distance between Robot {robot_index} and animal: {cos}")
                similarities.append(cos)
        except ValueError as e:
            print(f"Skipping animal for Robot {robot_index} due to shape mismatch.")

    return np.array(similarities)

def combination_fitnesses(distance,df_robot,df_animal,a,similarity_type):
    '''
    # For MSE DTW VAE +
    # For cosine -
    # distance[0,1] best:1  animal_similarity:[0,1] best:1
    # fitness[-1:0] best:-1
    '''
    # similarity_type = "MSE"
    print("similarity_type",similarity_type)
    # DTW_similarity
    if  similarity_type == "DTW":
        animal_similarity = calculate_similarity(df_robot,similarity_type)
        animal_similarity=-animal_similarity
        animal_similarity=DTW_fitness_scaling(animal_similarity)
    elif similarity_type=="Cosine":
    # Cosine_similarity
        animal_similarity = calculate_similarity(df_robot,similarity_type)
        animal_similarity=Cosine_fitness_scaling(animal_similarity)
    #MSE
    elif similarity_type=="MSE":
        animal_similarity = calculate_similarity(df_robot,similarity_type)
        animal_similarity=-animal_similarity
        animal_similarity=MSE_fitness_scaling(animal_similarity)

    #VAE
    elif similarity_type=="VAE":
        df_robot = infer_on_csv(df_robot)
        animal_similarity=VAE_similarity(df_robot,df_animal)
        animal_similarity = (-1) * np.array(animal_similarity)
        animal_similarity=VAE_fitness_scaling(animal_similarity)

    distance=distance_fitness_scaling(distance)
    combination=-a*np.array(distance)-(1-a)*np.array(animal_similarity)
    print("combined_fitnesses",combination)
    return combination,distance,animal_similarity

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
    print('Distance_fitnesses',fitnesses)
    print('Distance_scaled_fitnesses',scaled_fitness)
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
    print('DTW_fitnesses',fitnesses)
    print('DTW_scaled_fitnesses',scaled_fitness)
    return scaled_fitness

def Cosine_fitness_scaling(fitnesses):
    # best 1 worst 0
    scaled_fitness=[]
    min_f =-1
    max_f =1
    if min_f > np.min(fitnesses):
        min_f = np.min(fitnesses)
    if max_f < np.max(fitnesses):
        max_f = np.max(fitnesses)
    for i in fitnesses:
        scaled_fitness.append((i - min_f) / (max_f - min_f))
    print('Cosine_fitnesses',fitnesses)
    print('Cosine_scaled_fitnesses',scaled_fitness)
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
    print('VAE_fitnesses',fitnesses)
    print('VAE_scaled_fitnesses',scaled_fitness)
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
    print('MSE_fitnesses',fitnesses)
    print('MSE_scaled_fitnesses',scaled_fitness)
    return scaled_fitness