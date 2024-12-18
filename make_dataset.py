"""
Author: qp
Email: 3340593867@qq.com
"""
import numpy as np
import pandas as pd

# 设置随机种子
np.random.seed(42)

# 定义活动类别
activities = ['walk', 'run', 'sit', 'stand']

# 每个活动的持续时间（以时间点为单位）
samples_per_activity = 1000  # 可以根据需要调整

# 采样频率（Hz），假设每秒采样50次
sampling_rate = 50

# 时间序列
time = np.arange(samples_per_activity) / sampling_rate

data = []

for activity in activities:
    # 初始化加速度数组
    ax = np.zeros(samples_per_activity)
    ay = np.zeros(samples_per_activity)
    az = np.zeros(samples_per_activity)
    
    # 模拟加速度数据
    if activity == 'walk':
        # 模拟行走的加速度信号，使用低频正弦波加噪声
        frequency = 1.5  # 频率（Hz）
        ax = 0.5 * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, 0.1, samples_per_activity)
        ay = 0.4 * np.sin(2 * np.pi * frequency * time + np.pi/4) + np.random.normal(0, 0.1, samples_per_activity)
        az = 9.8 + 0.2 * np.sin(2 * np.pi * frequency * time + np.pi/2) + np.random.normal(0, 0.1, samples_per_activity)
    elif activity == 'run':
        # 模拟跑步的加速度信号，使用较高频率的正弦波加噪声
        frequency = 3.0  # 频率（Hz）
        ax = 1.0 * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, 0.2, samples_per_activity)
        ay = 0.8 * np.sin(2 * np.pi * frequency * time + np.pi/4) + np.random.normal(0, 0.2, samples_per_activity)
        az = 9.8 + 0.5 * np.sin(2 * np.pi * frequency * time + np.pi/2) + np.random.normal(0, 0.2, samples_per_activity)
    elif activity == 'sit':
        # 模拟坐着的加速度信号，基本为常数加小噪声
        ax = np.random.normal(0, 0.02, samples_per_activity)
        ay = np.random.normal(0, 0.02, samples_per_activity)
        az = 9.8 + np.random.normal(0, 0.05, samples_per_activity)
    elif activity == 'stand':
        # 模拟站立的加速度信号，与坐着类似，但噪声更小
        ax = np.random.normal(0, 0.01, samples_per_activity)
        ay = np.random.normal(0, 0.01, samples_per_activity)
        az = 9.8 + np.random.normal(0, 0.02, samples_per_activity)
    
    # 生成标签
    labels = np.array([activity] * samples_per_activity)
    
    # 将数据合并
    activity_data = np.column_stack((ax, ay, az, labels))
    data.append(activity_data)

# 将所有活动的数据按顺序连接，模拟连续采集的情况
data = np.vstack(data)

# 创建 DataFrame
df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'label'])

# 保存为 CSV 文件
df.to_csv('dataset/train_and_test_data.csv', index=False)
print('模拟的时间序列数据已保存到 dataset/train_and_test_data.csv')