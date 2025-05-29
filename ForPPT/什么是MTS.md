
根据您提供的代码，我来详细解释MTS data格式：

## 🔍 MTS Data格式解析

### **MTS = Multivariate Time Series（多变量时间序列）**

## 📊 数据结构

### **1. MTSData类的核心属性：**

```python
class MTSData:
    def __init__(self, train, valid, test, train_label, test_label, valid_label, info):
        self.train = train          # 训练集数据
        self.valid = valid          # 验证集数据  
        self.test = test            # 测试集数据
        self.train_label = train_label    # 训练集标签
        self.test_label = test_label      # 测试集标签
        self.valid_label = valid_label    # 验证集标签
        self.info = info            # 数据集信息
```

### **2. 数据维度说明：**

- **train/valid/test**: `shape = (时间步数, 特征数量)`
  - 例如：`(22784, 38)` 表示22784个时间步，每个时间步有38个特征
- **labels**: `shape = (时间步数,)` 
  - 二进制标签：0=正常，1=异常

## 🆚 MTS vs UTS 对比

| 特征 | **MTS (多变量)** | **UTS (单变量)** |
|------|------------------|------------------|
| **变量数量** | 多个特征 (如38个) | 单个特征 |
| **数据形状** | `(time_steps, features)` | `(time_steps,)` |
| **应用场景** | 机器监控、金融市场 | 温度、股价单指标 |
| **复杂度** | 高（需考虑特征间关系） | 低 |

## 🏭 实际例子：机器监控数据

以您的machine-1数据为例：

```python
# 假设shape为(22784, 38)
train_data = [
    # 时间步1: [温度, 压力, 振动, 电流, ..., 其他34个传感器]
    [25.3, 1.2, 0.05, 2.1, ..., sensor_38_value],
    # 时间步2:
    [25.5, 1.3, 0.06, 2.0, ..., sensor_38_value],
    # ...
    # 时间步22784:
    [26.1, 1.1, 0.04, 2.2, ..., sensor_38_value]
]

train_label = [0, 0, 0, 1, 1, 0, ...]  # 对应每个时间步的异常标签
```

## 🔧 数据预处理方法

MTSData提供了专门的预处理：

### **1. Z-score标准化：**
```python
def z_score_norm(self):
    # 对每个特征独立进行标准化
    for i in range(n_features):
        scaler = StandardScaler().fit(self.train[:, i])
        self.train[:, i] = scaler.transform(self.train[:, i])
```

### **2. MinMax归一化：**
```python
def min_max_norm(self, feature_range=(0,1)):
    # 对每个特征独立进行归一化到指定范围
```

## 📂 文件组织结构

```
datasets/
├── MTS/
│   ├── machine-1/
│   │   ├── train.npy      # 训练数据 (time_steps, features)
│   │   ├── train_label.npy # 训练标签 (time_steps,)
│   │   ├── test.npy       # 测试数据
│   │   ├── test_label.npy # 测试标签
│   │   └── info.json      # 元信息
```

## 🎯 为什么用MTS？

1. **多维监控**：同时监控多个传感器/指标
2. **关联分析**：捕获特征间的相互依赖
3. **全面诊断**：单一异常可能在多个维度体现
4. **提高精度**：多变量信息比单变量更丰富

## 💡 在TimeMixer中的应用

```python
# MTSData自动处理多变量特征
self.config.n_features = train_data.shape[1]  # 38个特征

# 模型需要处理多维输入
def forward(self, x):
    # x: (batch, seq_len, n_features) = (32, 128, 38)
    # 每个样本包含128个时间步，每个时间步38个特征
```

这就是为什么您的TimeMixer能够处理machine-1的38维传感器数据，进行多变量时间序列异常检测！
