import cv2

def put_pixel(image, y, x, radius=1):
    for i in range(y - radius, y + radius + 1):
        for j in range(x - radius, x + radius + 1):
            image[i, j] = [255, 0, 0]
    cv2.imwrite(f"output.png", image)

def print_zone(image, y, x, radius=4):
    for i in range(y - radius, y + radius + 1):
        for j in range(x - radius, x + radius + 1):
            print(image[i, j])

def video_to_frames():
    video_path = "./Files/gecko_360p _cutDLC_resnet50_GeckoPoseNov3shuffle1_190000_labeled.mp4"
    cap = cv2.VideoCapture(video_path)

    nr = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame[:, :240] = [0, 0, 0] # Delete the left part
        frame[:, 400:] = [0, 0, 0] # Delete the right part
        cv2.imwrite(f"./Files/Frames/frame_{nr}.png", frame)
        nr += 1

def get_points_from_frame(frame, num_points=13):
    points = []
    cursor_pos = [frame.shape[1] // 2, frame.shape[0] // 2]

    labels = [chr(ord('A') + i) for i in range(13)]
    print(f"Use arrow keys to move the cursor. Press Enter to select a point")
    print(f"Select {labels[len(points)]}")
    while True:
        temp_frame = frame.copy()

        # Draw previously selected points
        for p in points:
            cv2.circle(temp_frame, p, 5, (0, 255, 0), -1)

        # Draw cursor (black pixel)
        cv2.circle(temp_frame, tuple(cursor_pos), 3, (0, 0, 0), -1)

        cv2.imshow("Frame", temp_frame)
        key = cv2.waitKey(0)

        # Arrow keys
        if key == 81:  # Left arrow
            cursor_pos[0] = max(0, cursor_pos[0] - 1)
        elif key == 82:  # Up arrow
            cursor_pos[1] = max(0, cursor_pos[1] - 1)
        elif key == 83:  # Right arrow
            cursor_pos[0] = min(frame.shape[1] - 1, cursor_pos[0] + 1)
        elif key == 84:  # Down arrow
            cursor_pos[1] = min(frame.shape[0] - 1, cursor_pos[1] + 1)
        elif key == 10 or key == 13:  # Enter key (Linux: 10, Windows: 13)
            points.append(tuple(cursor_pos))
            if len(points) < num_points:
                print(f"Select {labels[len(points)]}")
            else:
                break

    cv2.destroyAllWindows()
    return points

