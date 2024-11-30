import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义自定义缺失值检测函数
def is_missing(value):
    return value.strip() == "('', '')"

# 计算每个列的识别率（包括排除joint列）
def calculate_column_detection_rate(csv_file_path, exclude_joint=True):
    data = pd.read_csv(csv_file_path)

    # 提取所有关键点列（假设第0列是帧号，其他列是关键点）
    keypoint_columns = data.columns[1:]  # 假设第 0 列是帧号，其他列是关键点

    # 如果排除特定列，则移除包含"center"、"forward"、"joint"的列
    if exclude_joint:
        exclude = ['center', 'forward', 'joint']
        keypoint_columns = [col for col in keypoint_columns if not any(excl in col.lower() for excl in exclude)]

    # 计算每个列的识别率
    detection_rates = {}
    for column in keypoint_columns:
        total_frames = len(data)  # 总帧数
        missing_frames = data[column].map(is_missing).sum()  # 缺失的帧数
        detected_frames = total_frames - missing_frames  # 检测到的帧数

        # 计算识别率
        detection_rate = detected_frames / total_frames
        detection_rates[column] = detection_rate

    return detection_rates

# 指定目标文件夹
folder_path = "./save_data/Experiment_set_1/"

# 存储所有文件的列识别率信息
all_detection_rates = {}

# 遍历文件夹中的所有 CSV 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # 只处理 CSV 文件
        file_path = os.path.join(folder_path, file_name)
        try:
            detection_rates = calculate_column_detection_rate(file_path, exclude_joint=True)
            for column, detection_rate in detection_rates.items():
                if column not in all_detection_rates:
                    all_detection_rates[column] = []
                all_detection_rates[column].append(detection_rate)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# 计算每个列（关键点）的平均识别率
average_detection_rates = {column: np.mean(rates) for column, rates in all_detection_rates.items()}

# 计算所有列的总体平均识别率
overall_average_detection_rate = np.mean(list(average_detection_rates.values()))

# 计算去掉 "joint" 列后的总体平均识别率
filtered_detection_rates = {column: rate for column, rate in average_detection_rates.items() if "joint" not in column.lower()}
overall_filtered_average_detection_rate = np.mean(list(filtered_detection_rates.values()))

# 转换为 DataFrame 方便显示
average_detection_rate_df = pd.DataFrame(list(average_detection_rates.items()), columns=["Column Name", "Average Detection Rate"])
filtered_detection_rate_df = pd.DataFrame(list(filtered_detection_rates.items()), columns=["Column Name", "Filtered Average Detection Rate"])

# 输出每个列的平均识别率和总体平均识别率
print("Average Detection Rate per Column (including all columns):")
print(average_detection_rate_df)
print("\nOverall Average Detection Rate (including all columns): ", overall_average_detection_rate)

print("\nFiltered Average Detection Rate per Column (excluding 'joint' columns):")
print(filtered_detection_rate_df)
print("\nOverall Filtered Average Detection Rate (excluding 'joint' columns): ", overall_filtered_average_detection_rate)

# # 绘制图形：每个列的平均识别率
# plt.figure(figsize=(10, 6))
# plt.bar(average_detection_rate_df['Column Name'], average_detection_rate_df['Average Detection Rate'], color='skyblue', label='All Columns')
# plt.ylabel('Average Detection Rate')
# plt.xlabel('Column Name')
# plt.title('Average Detection Rate per Column Across All Files')
# plt.xticks(rotation=90)  # 使列名显示更清晰
# plt.grid(True, axis='y')  # 只在 y 轴上显示网格
# plt.tight_layout()

# 绘制去掉 "joint" 列后的图形
plt.figure(figsize=(10, 6))
plt.bar(filtered_detection_rate_df['Column Name'], filtered_detection_rate_df['Filtered Average Detection Rate'], color='lightgreen', label='Filtered Columns (no "joint")')
plt.ylabel('Average Detection Rate')
plt.xlabel('Box')
plt.title('Average Detection Rate per box')
plt.xticks(rotation=90)  # 使列名显示更清晰
plt.grid(True, axis='y')  # 只在 y 轴上显示网格
plt.tight_layout()

# 显示图形
plt.show()

# # 绘制总体平均识别率
# plt.figure(figsize=(6, 4))
# plt.bar(["Overall"], [overall_average_detection_rate], color='salmon', label='Overall (All Columns)')
# plt.xlabel('Overall Average Detection Rate')
# plt.title('Overall Average Detection Rate Across All Columns')
# plt.tight_layout()
# plt.show()

# 绘制去掉 "joint" 列后的总体平均识别率
# plt.figure(figsize=(6, 4))
# plt.bar(["Overall Filtered"], [overall_filtered_average_detection_rate], color='lightcoral', label='Overall Filtered (no "joint")')
# plt.xlabel('Overall Average Detection Rate')
# plt.title('Overall Average Detection Rate (excluding "joint")')
# plt.tight_layout()
# plt.show()