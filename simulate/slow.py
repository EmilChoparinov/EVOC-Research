import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft


def slow_motion_fourier(data, scale_factor):
    """
    使用傅里叶变换放慢周期性运动数据。

    Args:
        data (np.ndarray): 周期性运动数据，形状为 (num_frames, 2)。
        scale_factor (int): 放慢倍数。

    Returns:
        np.ndarray: 放慢后的插值数据，形状为 (num_frames * scale_factor, 2)。
    """
    num_frames = data.shape[0]
    extended_length = num_frames * scale_factor
    result = np.zeros((extended_length, 2))

    for i in range(2):  # 分别处理 x 和 y
        original_signal = data[:, i]
        fft_coeffs = fft(original_signal)
        padded_fft = np.zeros(extended_length, dtype=complex)
        padded_fft[:num_frames // 2] = fft_coeffs[:num_frames // 2]
        padded_fft[-(num_frames // 2):] = fft_coeffs[-(num_frames // 2):]
        slowed_signal = np.real(ifft(padded_fft)) * scale_factor
        result[:, i] = slowed_signal

    return result


def slow_animal_with_linear_motion(scale_factor=4):
    """
    分离直线运动和周期性运动，并放慢周期性运动。
    Args:
        scale_factor (int): 放慢倍数，默认是 4 倍。
    """
    # 读取 CSV 文件
    file_path = './src/model/animal_data_head_orgin_884.csv'
    df = pd.read_csv(file_path)

    # 提取帧号
    frames = df['Frame'].values
    num_new_frames = len(frames) * scale_factor
    new_frames = np.linspace(frames.min(), frames.max(), num=num_new_frames)

    # 初始化结果
    interpolated_data = {'Frame': new_frames}

    # 关键点
    body_parts = ['middle', 'rear', 'left_front', 'left_hind', 'head', 'right_hind', 'right_front']

    # 计算整体运动：所有关键点的中心
    overall_motion = np.zeros((len(frames), 2))
    for part in body_parts:
        coords = df[part].str.strip('()').str.split(',', expand=True).astype(float)
        overall_motion += coords.values
    overall_motion /= len(body_parts)  # 平均位置即为整体运动

    # 插值整体运动（线性插值）
    interp_overall_x = interp1d(frames, overall_motion[:, 0], kind='linear', fill_value='extrapolate')
    interp_overall_y = interp1d(frames, overall_motion[:, 1], kind='linear', fill_value='extrapolate')
    overall_new_x = interp_overall_x(new_frames)
    overall_new_y = interp_overall_y(new_frames)

    for part in body_parts:
        # 提取关键点坐标
        coords = df[part].str.strip('()').str.split(',', expand=True).astype(float)
        keypoint_data = coords.values

        # 分离周期性运动
        periodic_motion = keypoint_data - overall_motion

        # 放慢周期性运动
        slowed_periodic_motion = slow_motion_fourier(periodic_motion, scale_factor)

        # 合成新轨迹
        new_x = slowed_periodic_motion[:, 0] + overall_new_x
        new_y = slowed_periodic_motion[:, 1] + overall_new_y

        # 格式化为 "(x, y)" 格式
        interpolated_data[part] = [f"({x:.2f}, {y:.2f})" for x, y in zip(new_x, new_y)]

    # 保存结果
    interpolated_df = pd.DataFrame(interpolated_data)
    output_file = f'./src/model/slow_with_linear_{scale_factor}.csv'
    interpolated_df.to_csv(output_file, index=False)

    print(f"Interpolated data saved to: {output_file}")

# def slow_animal(scale_factor=4):
#     """
#     放慢动物的运动。
#     Args:
#         scale_factor (int): 插值的倍数，4 表示帧数增加 4 倍。
#     """
#     file_path = './src/model/animal_data_head_orgin_884.csv'
#     df = pd.read_csv(file_path)
#
#     frames = df['Frame'].values
#
#     new_frames = np.linspace(frames.min(), frames.max(), num=(frames.max() - frames.min()) * scale_factor + 1)
#
#     interpolated_data = {'Frame': new_frames}
#
#     body_parts = ['middle', 'rear', 'left_front', 'left_hind', 'head', 'right_hind', 'right_front']
#
#     for part in body_parts:
#         coords = df[part].str.strip('()').str.split(',', expand=True).astype(float)
#         x_values = coords[0].values
#         y_values = coords[1].values
#
#         interp_x = interp1d(frames, x_values, kind='linear', fill_value='extrapolate')
#         interp_y = interp1d(frames, y_values, kind='linear', fill_value='extrapolate')
#
#
#         new_x = interp_x(new_frames)
#         new_y = interp_y(new_frames)
#
#         # 格式化为 "(x, y)" 格式
#         interpolated_data[part] = [f"({x:.2f}, {y:.2f})" for x, y in zip(new_x, new_y)]
#
#     interpolated_df = pd.DataFrame(interpolated_data)
#     output_file = f'./src/model/slow_interpolated_{scale_factor}.csv'
#     interpolated_df.to_csv(output_file, index=False)
#
#     print(f"Interpolated data saved to: {output_file}")



if __name__ == "__main__":
    slow_animal_with_linear_motion(scale_factor=3)