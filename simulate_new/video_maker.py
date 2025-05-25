import ast

import cv2
import numpy as np
import pandas as pd

from simulate_new.data import edge_definition, point_definition


def create_video_state(animal_behavior):
    frame_height = 720
    frame_width = 1080
    center_xy = (frame_width // 2, frame_height // 2)

    # Snap both to center across all rows
    def snap_to_center(row):
        center = row["middle"]
        for point in point_definition:
            row[point] = (row[point][0] - center[0], row[point][1] - center[1])
        return row

    # animal_behavior = animal_behavior.apply(snap_to_center, axis=1)

    animal_color = (0, 0, 0)

    video = cv2.VideoWriter(
        f"animal_walking.mp4",
        cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))

    for frame in range(len(animal_behavior)):
        frame_animal = animal_behavior.iloc[frame]

        # Create white background - fixed dimensions
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Add points
        for point in point_definition:
            # Add Animal Point
            coord = ast.literal_eval(frame_animal[point])
            screen_coord = (
                int(center_xy[0] + coord[0]),
                int(center_xy[1] - coord[1]) % frame_height
            )
            cv2.circle(img, screen_coord, 3, animal_color, -1)

        # Add Edges
        for p1, p2 in edge_definition:
            # Add Animal Edge
            coord_p1 = ast.literal_eval(frame_animal[p1])
            coord_p2 = ast.literal_eval(frame_animal[p2])

            screen_coord_p1 = (
                int(center_xy[0] + coord_p1[0]),
                int(center_xy[1] - coord_p1[1]) % frame_height
            )
            screen_coord_p2 = (
                int(center_xy[0] + coord_p2[0]),
                int(center_xy[1] - coord_p2[1]) % frame_height
            )

            cv2.line(img, screen_coord_p1, screen_coord_p2, animal_color, 2)

        video.write(img)

    video.release()
    cv2.destroyAllWindows()

create_video_state(pd.read_csv("./Files/animal_data_3_slow_down_lerp_2.csv"))
