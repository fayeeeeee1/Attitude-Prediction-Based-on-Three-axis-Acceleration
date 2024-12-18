# Attitude-Prediction-Based-on-Three-axis-Acceleration

A simple demo, using accelerometer data, classifies and recognizes different activities (e.g. walking, running, sitting, standing) by building convolutional neural networks and long short-term memory network (CNN-LSTM) models. The project covers the complete process of data generation, preprocessing, model training, prediction, and more.

## 项目结构

```
├── dataset
│   ├── train_and_test_data.csv       # 训练和测试数据集
│   └── verify_data.csv               # 验证数据集
├── results
│   └── predictions.csv               # 预测结果
├── runs
│   ├── scaler.save                   # 标准化器
│   ├── label_encoder.save            # 标签编码器
│   └── trained_model.h5              # 训练好的模型
├── main.py                           # 主程序（可选）
├── make_dataset.py                   # 生成训练和测试数据集的脚本
├── make_verify_dataset.py            # 生成验证数据集的脚本
├── train.py                          # 模型训练脚本
├── predict.py                        # 模型预测脚本
└── README.md                         # 项目说明文件
```

## 环境依赖

请确保已安装以下 Python 包：

- numpy
- pandas
- scikit-learn
- tensorflow
- keras
- joblib

使用以下命令安装依赖项：

```bash
pip install numpy pandas scikit-learn tensorflow keras joblib
```

## 数据集生成

### 生成训练和测试数据集

运行以下命令生成模拟的训练和测试数据，并保存在 `train_and_test_data.csv`文件中：

```bash
python make_dataset.py
```

### 生成验证数据集

运行以下命令生成模拟的验证数据，并保存在 `verify_data.csv`文件中：

```bash
python make_verify_dataset.py
```

## 模型训练

运行以下命令进行模型训练：

```bash
python train.py
```

**步骤概览：**

1. **加载数据**：从 `train_and_test_data.csv`文件中加载加速度数据和标签。

2. **数据预处理**：
   - 对加速度数据进行标准化处理。
   - 使用 `LabelEncoder`将活动标签转换为数字编码。

3. **数据窗口化**：
   - 使用滑动窗口方法将时间序列数据转换为适合模型输入的形状。
   - 窗口大小为 100，滑动步长为 50。

4. **构建模型**：
   - 使用 `Sequential`构建`CNN-LSTM`模型，包括卷积层、池化层、LSTM 层、全连接层等。

5. **模型训练**：
   - 编译模型并在训练集上进行训练，验证集用于评估模型性能。

6. **模型评估和保存**：
   - 在测试集上评估模型的准确率。
   - 保存训练好的模型、标准化器和标签编码器到 `runs`目录。

## 模型预测

运行以下命令使用训练好的模型对新数据进行预测：

```bash
python predict.py
```

**步骤概览：**

1. **加载模型和预处理工具**：
   - 加载保存的标准化器、标签编码器和模型。

2. **加载新数据**：
   - 从 `verify_data.csv`

 文件中加载需要进行预测的加速度数据。

3. **数据预处理**：
   - 对新数据进行与训练数据相同的标准化处理。
   - 进行数据窗口化以匹配模型输入形状。

4. **进行预测**：
   - 使用模型对预处理后的数据进行预测。
   - 将预测的数字标签转换为原始活动名称。

5. **结果输出**：
   - 在终端显示每个样本的预测标签。
   - 将预测结果保存为 `predictions.csv`


## 使用说明

1. **克隆或下载本项目代码。**

2. **安装依赖环境：**

   ```bash
   pip install -r requirements.txt
   ```

3. **生成数据集：**

   ```bash
   python make_dataset.py
   python make_verify_dataset.py
   ```

4. **训练模型：**

   ```bash
   python train.py
   ```

5. **进行预测：**

   ```bash
   python predict.py
   ```

## 示例结果

在训练完成后，模型将在测试集上输出准确率，例如：

```
测试集准确率: 99.85%
```

再运行 

`predict.py`

 后，预测结果将在终端显示，并保存到 

`predictions.csv`

例如：

```
样本 1 的预测标签: walk
样本 2 的预测标签: run
...
```
