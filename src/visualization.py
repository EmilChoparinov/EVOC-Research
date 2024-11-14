import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from rotation_scaling import translation_rotation, size_scaling, translation_rotation_csv, size_scaling_csv
import cv2
import ast


def plot_distance_similarity(distance_all, animal_similarity_all):
    # Calculate max distance and similarity values
    max_distances = [np.min(distance) for distance in distance_all]
    # if there is no scale up, we should add shirnk 0.000001
    max_similarities = [np.min(similarity) for similarity in animal_similarity_all]

    plt.figure(figsize=(10, 6))
    plt.scatter(max_similarities, max_distances, color='blue', marker='o', label="Distance vs Animal Similarity")

    plt.title("Fitness Plot: Animal Similarity vs Distance")
    plt.xlabel("Animal Similarity (Max Value)")
    plt.ylabel("Distance (Max Value)")
    plt.legend()
    plt.show()

def plot_distance_fitness(distance_all):
    max_distances = [np.min(distance) for distance in distance_all]
    print("Max distances:", max_distances)
    # print("Max similarities:", max_similarities)
    iterations = np.arange(1, len(distance_all) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, max_distances, label="Distance", color='blue', marker='o')

    plt.title("Fitness Plot: Distance")
    plt.xlabel("Generation")
    plt.ylabel("Max Value")
    plt.legend()
    plt.show()

def plot_fitness(distance_all, animal_similarity_all):
    # max_distances = [(-1)*np.min(distance) for distance in distance_all]
    max_similarities = [np.min(similarity) for similarity in animal_similarity_all]

    # sum_values = [0.6*max_distance+0.4*max_similarity for max_distance, max_similarity in
    #               zip(max_distances, max_similarities)]
    # print("Max distances:", max_distances)
    print("Max similarities:", max_similarities)
    iterations = np.arange(1, len(distance_all) + 1)

    plt.figure(figsize=(10, 6))

    # plt.plot(iterations, max_distances, label="Distance", color='blue', marker='o')

    plt.plot(iterations, max_similarities, label="Animal Similarity", color='green', marker='x')

    # plt.plot(iterations, sum_values, label="Sum of Distance & Animal Similarity", color='red', marker='s')
    plt.title("Fitness Plot: Animal Similarity")
    plt.xlabel("Generation")
    plt.ylabel("Max Value")
    plt.legend()
    plt.show()

# visualize single video
def visualization_simulation(scaled_df: pd.DataFrame, save_path: str = './video/scaled_robot.mp4'):
    """
    Creates a video visualizing the scaled and rotated positions of each robot's body parts.

    Parameters:
        scaled_df (pd.DataFrame): DataFrame containing scaled and rotated coordinates.
        save_path (str): Path to save the generated video.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_title("Transformed Robot Motion")

    # Plot elements for transformed points
    transformed_points, = ax.plot([], [], 'ro', label='Transformed', markersize=5)
    ax.legend(loc="upper right")

    def init():
        transformed_points.set_data([], [])
        return transformed_points,

    def animate(frame):
        # Extract points for the current frame
        transformed_row = scaled_df[scaled_df['Frame'] == frame]

        # Body parts to visualize
        body_parts = ['head', 'middle', 'rear', 'right_front', 'left_front', 'right_hind', 'left_hind']

        # Transformed points
        transformed_x = [transformed_row[part].values[0][0] for part in body_parts]
        transformed_y = [transformed_row[part].values[0][1] for part in body_parts]

        # Update data for plotting
        transformed_points.set_data(transformed_x, transformed_y)

        return transformed_points,

    # Create the animation
    frames = scaled_df['Frame'].max() + 1  # Total number of frames
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=100, blit=True)

    # Save the animation as an mp4 file
    ani.save(save_path, writer='ffmpeg', fps=10)
    plt.show()

# visualize two csv
def visualization_csv(scaled_df: pd.DataFrame, scaled_df_2: pd.DataFrame, save_path: str = 'simulation_pair.mp4'):
    columns_to_plot = {
        'head': (255, 0, 0),
        'middle': (0, 255, 0),
        'rear': (0, 0, 255),
        'left_front': (255, 255, 0),
        'right_front': (255, 0, 255),
        'left_hind': (0, 255, 255),
        'right_hind': (128, 0, 128),
    }

    frames = scaled_df['Frame'].unique()
    frame_width, frame_height = 1080, 960
    fps = 10

    center_x, center_y = frame_width // 2, frame_height // 2

    output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame in frames:
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        frame_data = scaled_df[scaled_df['Frame'] == frame]
        frame_data_2 = scaled_df_2[scaled_df_2['Frame'] == frame]

        coords = {}
        coords_2 = {}

        for column, color in columns_to_plot.items():
            if column in frame_data.columns:  # check the column is existing
                for index, row in frame_data.iterrows():
                    coord = row[column]
                    if isinstance(coord, tuple) and len(coord) == 2:
                        x = int(center_x + coord[0])
                        y = int(center_y - coord[1])
                        cv2.circle(img, (x, y), 3, color, -1)
                        coords[column] = (x, y)
                        if column == 'right_hind':
                            cv2.putText(img, 'right_hind', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                        cv2.LINE_AA)
                        elif column == 'right_front':
                            cv2.putText(img, 'right_front', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                        cv2.LINE_AA)
                    else:
                        print(f"Skipping {column} in frame {frame} due to invalid format: {coord}")

            if column in frame_data_2.columns:
                for index, row in frame_data_2.iterrows():
                    coord_2 = row[column]
                    if isinstance(coord_2, tuple) and len(coord_2) == 2:
                        x = int(center_x + coord_2[0])
                        y = int(center_y - coord_2[1])
                        cv2.circle(img, (x, y), 3, color, -1)
                        coords_2[column] = (x, y)
                        # if column == 'right_hind':
                        #     cv2.putText(img, 'right_hind', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        #                 cv2.LINE_AA)
                        # elif column == 'right_front':
                        #     cv2.putText(img, 'right_front', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        #                 cv2.LINE_AA)
                    else:
                        print(f"Skipping {column} in frame {frame} due to invalid format: {coords_2}")



        connections = [
            ('head', 'left_front'), ('head', 'right_front'), ('head', 'middle'),
            ('middle', 'rear'), ('rear', 'left_hind'), ('rear', 'right_hind')
        ]

        for part1, part2 in connections:
            # best_generation(red)
            if part1 in coords and part2 in coords:
                cv2.line(img, coords[part1], coords[part2], (0, 0, 255), 1)
            # first generation (black)
            if part1 in coords_2 and part2 in coords_2:
                cv2.line(img, coords_2[part1], coords_2[part2], (0, 0, 0), 1)

        output_video.write(img)

    output_video.release()
    print(f"video saved under {save_path}")

def visualize_run():
    # after running, you can read the most-fitxy-run-X.csv udner src
    df = pd.read_csv('most-fit-xy-run-1.csv')

    # print(generation)
    max_fitness_generation_id = df.loc[df['generation_best_fitness_score'].idxmin(), 'generation_id']
    df_generation = df[df['generation_id'] == max_fitness_generation_id]
    print(max_fitness_generation_id)
    # print(df_generation)
    df_first_generation = df[df['generation_id'] == 1]

    rotated_data_generation = translation_rotation_csv(df_generation)
    scale_data_generation = size_scaling_csv(rotated_data_generation)

    rotated_data_first_generation = translation_rotation_csv(df_first_generation)
    scale_data_first_generation = size_scaling_csv(rotated_data_first_generation)
    visualization_csv(scale_data_generation, scale_data_first_generation)


# visualize_run()