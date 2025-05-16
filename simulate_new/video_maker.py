import copy

import cv2
import numpy as np
import pandas as pd

from simulate_new import ea, stypes
from simulate_new.data import edge_definition, point_definition, convert_tuple_columns


def create_video_state(state: stypes.EAState):
    frame_height = 720
    frame_width = 1080
    center_xy = (frame_width // 2, frame_height // 2)

    last_gen = state.generation - 1

    robot_behavior = convert_tuple_columns(pd.read_csv(
        ea.file_idempotent(state, "Distance")).query(f"generation == {last_gen}"))
    animal_behavior = copy.deepcopy(state.animal_data)

    # Snap both to center across all rows
    def snap_to_center(row):
        center = row["middle"]
        for point in point_definition:
            row[point] = (row[point][0] - center[0], row[point][1] - center[1])
        return row

    robot_behavior = robot_behavior.apply(snap_to_center, axis=1)
    animal_behavior = animal_behavior.apply(snap_to_center, axis=1)

    robot_color = (0, 0, 255)
    animal_color = (0, 0, 0)

    video = cv2.VideoWriter(
        f"{ea.file_idempotent(state)}.mp4",
        cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

    frame_count = min(len(robot_behavior), len(animal_behavior))
    robot_behavior = robot_behavior.iloc[:frame_count]
    animal_behavior = animal_behavior.iloc[:frame_count]

    for frame in range(frame_count):
        frame_robot = robot_behavior.iloc[frame]
        frame_animal = animal_behavior.iloc[frame]

        # Create white background - fixed dimensions
        img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Add description of video
        cv2.putText(img,
                    f"{state.similarity_type} @ Alpha {state.alpha}",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img,
                    f"{frame}/{frame_count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

        # Add points
        for point in point_definition:
            # Add Robot Point
            coord = frame_robot[point]
            screen_coord = (
                int(center_xy[0] + coord[0]),
                int(center_xy[1] - coord[1])
            )
            cv2.circle(img, screen_coord, 3, robot_color, -1)

            # Add Animal Point
            coord = frame_animal[point]
            screen_coord = (
                int(center_xy[0] + coord[0]),
                int(center_xy[1] - coord[1])
            )
            cv2.circle(img, screen_coord, 3, animal_color, -1)

        # Add Edges
        for p1, p2 in edge_definition:
            # Add Robot Edge
            coord_p1 = frame_robot[p1]
            coord_p2 = frame_robot[p2]

            screen_coord_p1 = (
                int(center_xy[0] + coord_p1[0]),
                int(center_xy[1] - coord_p1[1])
            )
            screen_coord_p2 = (
                int(center_xy[0] + coord_p2[0]),
                int(center_xy[1] - coord_p2[1])
            )

            cv2.line(img, screen_coord_p1, screen_coord_p2, robot_color, 2)

            # Add Animal Edge
            coord_p1 = frame_animal[p1]
            coord_p2 = frame_animal[p2]

            screen_coord_p1 = (
                int(center_xy[0] + coord_p1[0]),
                int(center_xy[1] - coord_p1[1])
            )
            screen_coord_p2 = (
                int(center_xy[0] + coord_p2[0]),
                int(center_xy[1] - coord_p2[1])
            )

            cv2.line(img, screen_coord_p1, screen_coord_p2, animal_color, 2)

        video.write(img)

    video.release()
    cv2.destroyAllWindows()
