"""
  A
BCDEF
  G
HIJKL
  M
"""
import ast
import pandas as pd

from animal_video_to_animal_csv.util import slow_down_the_animal


def move_the_first_frame_of_D_in_origin(file_path="./Files/animal_data_0.csv"):
    # The animal moves in the second coordinate from 0 to -229
    # The robot moves in the second coordinate from 0 to +something (around +3)
    # So we reverse the second coordinate of the animal
    def subtract_translation(point_str):
        x, y = ast.literal_eval(point_str)
        return f"({x - translation[0]}, {translation[1] - y})"

    df = pd.read_csv(file_path)
    translation = ast.literal_eval(df.loc[0, "D"])
    for col in df.columns:
        if col != "Frame":
            df[col] = df[col].apply(subtract_translation)

    df.to_csv("./Files/animal_data_1.csv", index=False)
    # Now the animal moves in the second coordinate from 0 to 229

def refactor_animal_data(file_path="./Files/animal_data_1.csv"):
    df = pd.read_csv(file_path)
    df.drop(columns=["A", "B", "F", "H", "L", "M"], inplace=True)
    df.rename(columns={"C": "left_front", "D": "head", "E": "right_front",
                       "G": "middle", "I": "left_hind", "J": "rear",
                       "K": "right_hind"}, inplace=True)

    df.to_csv("./Files/animal_data_2.csv", index=False)

def extend_animal_data(file_path="./Files/animal_data_2.csv"):
    def add_points(p1, p2):
        return f"({p1[0] + p2[0]}, {p1[1] + p2[1]})"

    def interpolation(frame1, frame2):
        interp = frame1.copy()
        for col in interp.columns:
            if col != "Frame":
                p1 = ast.literal_eval(frame1.iloc[0][col])
                p2 = ast.literal_eval(frame2.iloc[0][col])
                mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                interp.at[frame1.index[0], col] = f"({mid[0]}, {mid[1]})"
        return interp

    original_df = pd.read_csv(file_path)
    last_frame = original_df.iloc[-1]
    translation = {
        col: ast.literal_eval(last_frame["head"])
        for col in original_df.columns if col != "Frame"
    }

    dfs = [original_df]
    for _ in range(7):
        df = dfs[-1].copy()
        for col in df.columns:
            if col != "Frame":
                df[col] = df[col].apply(
                    lambda p: add_points(ast.literal_eval(p), translation[col])
                )
        frame_1 = dfs[-1].iloc[[-1]].copy()
        frame_2 = df.iloc[[0]].copy()
        transition_frame = interpolation(frame_1, frame_2)

        dfs.append(transition_frame)
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df["Frame"] = range(len(result_df))
    result_df.to_csv("./Files/animal_data_3.csv", index=False)


#move_the_first_frame_of_D_in_origin()
refactor_animal_data()
extend_animal_data()
slow_down_the_animal(how_much=3)
