import numpy as np
import pandas as pd


def interpolate_points(start, end, num_points):
    start = np.array([float(x) for x in start.strip('()').split(',')])
    end = np.array([float(x) for x in end.strip('()').split(',')])
    return [tuple(start + (end - start) * (i / (num_points + 1))) for i in range(1, num_points + 1)]

def slow_down_the_animal(file_path_in="./Files/animal_data_3.csv", how_much=2):
    file_path_out = file_path_in.replace(".csv", f"_slow_down_lerp_{how_much}.csv", 1)
    data = pd.read_csv(file_path_in)
    new_data = []

    # Number of new frames between each original frame (5 times slower = 4 new frames)
    num_interpolated_frames = how_much - 1
    for i in range(len(data) - 1):
        new_data.append([data['Frame'][i]] + [data[col][i] for col in data.columns[1:]])

        for j in range(1, num_interpolated_frames + 1):
            frame = data['Frame'][i] + j * (data['Frame'][i + 1] - data['Frame'][i]) / (num_interpolated_frames + 1)
            interpolated_row = [frame]

            for col in data.columns[1:]:
                start_value = data[col][i]
                end_value = data[col][i + 1]
                interpolated_values = interpolate_points(start_value, end_value, num_interpolated_frames)
                interpolated_row.append(str(interpolated_values[j - 1]))

            new_data.append(interpolated_row)

    new_data.append([data['Frame'][len(data) - 1]] + [data[col][len(data) - 1] for col in data.columns[1:]])

    new_df = pd.DataFrame(new_data, columns=data.columns)
    new_df["Frame"] = range(len(new_df))
    new_df.to_csv(file_path_out, index=False)