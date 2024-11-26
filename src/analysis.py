import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def average_and_std_plot(similarity_type_to_plot='DTW'):
    """
    计算不同文件夹下的数据，并绘制以 Alpha 为区分的平均值和标准差图。

    Args:
        base_dir (str): 包含所有 results-{alpha}-{similarity_type} 文件夹的主目录。
        similarity_type_to_plot (str): 要绘制的相似性类型，例如 'DTW', 'MSE', 'Cosine'。
    """
    # 获取所有以 "results-{alpha}-{similarity_type}" 命名的文件夹
    base_dir=''
    result_dirs = glob.glob(os.path.join(base_dir, f'results-*-{similarity_type_to_plot}'))

    if not result_dirs:
        print(f"No result directories found in {base_dir} for similarity type: {similarity_type_to_plot}")
        return

    plt.figure(figsize=(10, 8))
    # 遍历每个文件夹
    for result_dir in result_dirs:
        # 提取 alpha 值（从文件夹名解析，例如: results-{alpha}-{similarity_type}）
        folder_name = os.path.basename(result_dir)
        parts = folder_name.split('-')
        alpha = float(parts[1])  # 第二部分是 alpha

        # 获取文件夹内所有 CSV 文件
        csv_files = glob.glob(os.path.join(result_dir, '*.csv'))
        if not csv_files:
            print(f"No CSV files found in directory: {result_dir}")
            continue

        all_distances = []
        all_similarities = []

        # 读取每个文件的数据
        for file in csv_files:
            df = pd.read_csv(file)
            all_distances.extend(df['Distance'].values)
            all_similarities.extend(df['Animal Similarity'].values)

        # 计算均值和标准差
        mean_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        mean_similarity = np.mean(all_similarities)
        std_similarity = np.std(all_similarities)

        # 绘制带误差的点图
        plt.errorbar(mean_similarity, mean_distance,
                     xerr=std_similarity, yerr=std_distance,
                     fmt='o', label=f'Alpha = {alpha}')

    plt.xlabel('Animal Similarity ')
    plt.ylabel('Distance ')
    plt.title(f'Average Distance vs Animal Similarity\n(Similarity Type: {similarity_type_to_plot})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # 使用 argparse 获取命令行参数
    parser = argparse.ArgumentParser(description="Plot Average Distance vs Animal Similarity.")

    parser.add_argument(
        "--similarity_type", type=str, default="DTW",
        help="Type of similarity to plot (e.g., 'DTW', 'MSE', 'Cosine')."
    )

    args = parser.parse_args()

    # 调用主函数
    average_and_std_plot(args.similarity_type)