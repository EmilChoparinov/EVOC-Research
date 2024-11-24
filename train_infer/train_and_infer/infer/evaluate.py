import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义自定义缺失值检测函数
def is_missing(value):
    return value.strip() == "('', '')"

# 修改识别率计算函数
def calculate_detection_rate(csv_file_path):
    data = pd.read_csv(csv_file_path)

    # 提取关键点列
    keypoint_columns = data.columns[1:]  # 假设第 0 列是帧号，其他列是关键点

    # 检测关键点是否缺失
    total_keypoints = len(keypoint_columns) * len(data)  # 总关键点数量
    missing_keypoints = data[keypoint_columns].map(is_missing).sum().sum()  # 缺失关键点数量
    detected_keypoints = total_keypoints - missing_keypoints  # 成功检测的关键点数量

    # 计算识别率
    detection_rate = detected_keypoints / total_keypoints
    return detection_rate, detected_keypoints, total_keypoints

# 指定目标文件夹
folder_path = "./save_data/Experiment_set_1/"

# 遍历文件夹中的所有 CSV 文件
detection_rate_summary = []  # 存储每个文件的识别率信息
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # 只处理 CSV 文件
        file_path = os.path.join(folder_path, file_name)
        try:
            detection_rate, detected, total = calculate_detection_rate(file_path)
            detection_rate_summary.append({
                "File Name": file_name,
                "Detection Rate": detection_rate,
                "Detected Keypoints": detected,
                "Total Keypoints": total
            })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# 转换为 DataFrame
detection_rate_df = pd.DataFrame(detection_rate_summary)

# 显示结果
print(detection_rate_df)