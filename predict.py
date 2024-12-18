"""
Author: qp
Email: 3340593867@qq.com
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# 1. 加载保存的 scaler 和 label_encoder
scaler = joblib.load('runs/scaler.save')
label_encoder = joblib.load('runs/label_encoder.save')
print('标准化器和标签编码器已加载')

# 2. 加载训练好的模型
model = load_model('runs/trained_model.h5')
print('模型已加载')

# 3. 加载新数据
# 请确保新数据的格式与训练数据一致，即包含 'ax', 'ay', 'az' 列
new_data = pd.read_csv('dataset/verify_data.csv') 

# 4. 确保数据类型正确
new_data['ax'] = new_data['ax'].astype(float)
new_data['ay'] = new_data['ay'].astype(float)
new_data['az'] = new_data['az'].astype(float)

# 5. 数据预处理
# 提取特征
X_new_raw = new_data[['ax', 'ay', 'az']].values

# 标准化
X_new_scaled = scaler.transform(X_new_raw)

# 6. 数据窗口化
window_size = 100  # 与训练时相同
step = 50

def create_windows_predict(data, window_size=100, step=50):
    num_samples = ((len(data) - window_size) // step) + 1
    X = np.zeros((num_samples, window_size, data.shape[1]))
    
    for i in range(num_samples):
        start = i * step
        end = start + window_size
        X[i] = data[start:end]
    return X

X_new = create_windows_predict(X_new_scaled, window_size, step)

# 7. 进行预测
predictions = model.predict(X_new)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# 8. 显示预测结果
for i, label in enumerate(predicted_labels):
    print(f'样本 {i+1} 的预测标签: {label}')

# 9. 保存预测结果为 CSV
predictions_df = pd.DataFrame({
    'sample': np.arange(1, len(predicted_labels) + 1),
    'predicted_label': predicted_labels
})

predictions_df.to_csv('results/predictions.csv', index=False)
print('预测结果已保存到 results/predictions.csv')