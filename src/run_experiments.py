import os
import subprocess

# 定义alpha和fitness function的组合
alpha_values = [1, 0.5, 0]
fitness_functions = ["distance", "similarity", "blended"]

# 循环遍历每种组合
for alpha in alpha_values:
    for fit in fitness_functions:
        print(f"Running with alpha={alpha} and fitness function={fit}")

        # 执行run.py文件，并传入参数
        result = subprocess.run(
            [
                "python3", "run.py",
                "--alpha", str(alpha),
                "--with-fit", fit,
                "--gens", "1",
                "--runs", "5",
                "--cleanup"  # 如果需要清理旧的CSV文件
            ],
            capture_output=True,
            text=True
        )

        # 打印输出结果用于调试
        print(result.stdout)
        print(result.stderr)

        # 检查是否运行成功
        if result.returncode != 0:
            print(f"Run failed with alpha={alpha} and fitness function={fit}")
