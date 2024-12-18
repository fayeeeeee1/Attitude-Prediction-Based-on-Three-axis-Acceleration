"""
Author: qp
Email: 3340593867@qq.com
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import joblib

# 设置随机种子
np.random.seed(42)

"""
数据格式:
ax,ay,az,label
0.1,0.2,-0.3,walk
0.5,0.4,0.1,walk
...
"""

# 1. 加载数据
data = pd.read_csv('dataset/train_and_test_data.csv') 

# 确保数据类型正确
data['ax'] = data['ax'].astype(float)
data['ay'] = data['ay'].astype(float)
data['az'] = data['az'].astype(float)

# 2. 数据预处理
# 提取特征和标签
X_raw = data[['ax', 'ay', 'az']].values
y_raw = data['label'].values

# 对加速度数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 将标签转换为数字编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# 将数据窗口化
def create_windows(data, labels, window_size=100, step=50):
    num_samples = ((len(data) - window_size) // step) + 1
    X = np.zeros((num_samples, window_size, data.shape[1]))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        start = i * step
        end = start + window_size
        X[i] = data[start:end]
        
        # 检查窗口内的标签是否一致
        window_labels = labels[start:end]
        if np.all(window_labels == window_labels[0]):
            y[i] = window_labels[0]
        else:
            # 如果窗口内存在不同的标签，选择出现次数最多的标签
            counts = np.bincount(window_labels)
            y[i] = np.argmax(counts)
    return X, y

window_size = 100  # 每个窗口包含 100 个时间点
step = 50  # 滑动步长
X, y = create_windows(X_scaled, y_encoded, window_size, step)

# 将标签转换为独热编码
y_categorical = to_categorical(y)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

# 3. 构建模型
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 5. 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集准确率: {accuracy * 100:.2f}%')

# 6. 保存模型
model.save('runs/trained_model.h5')
print('模型已保存到 runs/trained_model.h5')

# 保存 scaler
joblib.dump(scaler, 'runs/scaler.save')
print('标准化器已保存到 runs/scaler.save')

# 保存 label_encoder
joblib.dump(label_encoder, 'runs/label_encoder.save')
print('标签编码器已保存到 runs/label_encoder.save')