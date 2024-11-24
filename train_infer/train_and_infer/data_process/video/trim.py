import cv2


def trim_video(input_path, output_path, duration=25):
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(duration * fps)  # 前 10 秒的帧数

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器和输出视频对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 读取和保存前 10 秒的视频帧
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    # 释放资源
    cap.release()
    out.release()


# 使用例子
input_video = "Screen Recording 2024-11-06 at 02.23.38.mov"
output_video = "output_10_seconds.mp4"
trim_video(input_video, output_video)