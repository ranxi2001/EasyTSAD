# MTSMixer多元时序异常检测性能分析报告

## 📊 实验结果概览

### 当前MTSMixer性能指标
| 评估协议 | F1-Score | Precision | Recall |
|---------|----------|-----------|--------|
| Point-wise F1 PA | **0.823** | 0.871 | 0.837 |
| Event-based F1 PA (log) | 0.542 | 0.675 | 0.493 |
| Event-based F1 PA (squeeze) | 0.435 | 0.698 | 0.386 |

### 基准对比
- **同学简单模型**: 三个指标均达到 **80%+**
- **我们的MTSMixer**: Point-wise 82.3%，Event-based 54.2%/43.5%

## 🔍 问题分析

### 1. 模型设计不匹配问题
**问题**: MTSMixer原本为**时序预测**任务设计，而非异常检测
- **预测任务**: 关注序列的趋势和周期性模式
- **异常检测**: 关注偏离正常模式的微小变化

**影响**:
- 模型可能过度关注全局趋势，忽略局部异常
- 因子化混合可能平滑了异常信号

### 2. 模型复杂度过高
**问题**: 相比同学的"基础MTS+两层"方案，MTSMixer过于复杂

**复杂度对比**:
```
简单方案: MTS基础模型 + 2层神经网络
我们方案: RevIN + FactorizedTemporalMixing + FactorizedChannelMixing + MixerBlocks
```

**负面影响**:
- 参数过多导致过拟合
- 训练难度增大
- 对小数据集不友好

### 3. 重构策略问题
**当前策略**: 使用滑动窗口进行序列重构
```python
# 计算重构误差
reconstruction_error = torch.mean((window_batch - reconstructed) ** 2, dim=2)
```

**潜在问题**:
- 均方误差可能不敏感于异常模式
- 窗口大小(96)可能不适合异常检测
- 缺乏对时间局部性的考虑

### 4. 训练策略不当
**问题点**:
- **训练目标**: 最小化重构误差，但正常数据重构好≠异常检测好
- **数据使用**: 只用正常数据训练，缺乏对异常模式的理解
- **损失函数**: MSE损失对异常点不够敏感

### 5. Event-based性能差的原因
**Point-wise vs Event-based表现差异巨大**:
- Point-wise: 82.3%
- Event-based: 54.2%/43.5%

**原因分析**:
- 模型产生**散点式异常检测**，而非连续事件检测
- 缺乏**时序连续性建模**
- **后处理策略**不足，无法将点异常聚合为事件异常

## 💡 改进建议

### 1. 模型简化方案
```python
class SimpleMTSDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # 简化版本：只保留核心组件
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim)
        )
```

### 2. 异常检测专用损失函数
```python
def anomaly_detection_loss(original, reconstructed):
    # 组合多种损失
    mse_loss = F.mse_loss(original, reconstructed)
    mae_loss = F.l1_loss(original, reconstructed)
    
    # 对大误差进行惩罚
    threshold = torch.quantile(mse_loss, 0.95)
    penalty = torch.where(mse_loss > threshold, mse_loss * 2, mse_loss)
    
    return mae_loss + penalty.mean()
```

### 3. 时间窗口优化
- **多尺度窗口**: 使用多个窗口大小(24, 48, 96)
- **重叠窗口**: 增加窗口重叠度提高检测连续性
- **自适应窗口**: 根据数据特性调整窗口大小

### 4. Event-based后处理
```python
def point_to_event_postprocess(scores, threshold=0.5, min_event_length=3):
    """将点异常转换为事件异常"""
    binary_scores = scores > threshold
    
    # 连接相邻的异常点
    events = []
    start = None
    for i, is_anomaly in enumerate(binary_scores):
        if is_anomaly and start is None:
            start = i
        elif not is_anomaly and start is not None:
            if i - start >= min_event_length:
                events.append((start, i-1))
            start = None
    
    return events
```

## 🚀 快速改进方案

### 方案A: 轻量级MTSMixer
1. **减少层数**: e_layers=1
2. **简化混合**: 仅使用TemporalMixing，去除ChannelMixing
3. **小模型**: d_model=64, d_ff=128

### 方案B: 专用异常检测器
1. **双分支结构**: 一支重构，一支预测
2. **对比学习**: 正常vs重构的表示学习
3. **多任务学习**: 重构+异常分类

### 方案C: 集成简单模型
```python
class EnsembleDetector:
    def __init__(self):
        self.models = [
            SimpleAutoEncoder(),
            IsolationForest(),
            LOF()
        ]
    
    def detect(self, data):
        scores = [model.score(data) for model in self.models]
        return np.mean(scores, axis=0)
```

## 📈 预期改进效果

| 改进方案 | 预期Point F1 | 预期Event F1 | 复杂度 |
|---------|-------------|-------------|--------|
| 轻量级MTSMixer | 85%+ | 75%+ | 中等 |
| 专用异常检测器 | 88%+ | 80%+ | 高 |
| 集成简单模型 | 83%+ | 78%+ | 低 |

## 🎯 结论与建议

1. **立即行动**: 实现轻量级MTSMixer方案，快速验证改进效果
2. **中期目标**: 开发专用异常检测架构
3. **长期规划**: 建立异常检测模型库，支持多种场景

**核心问题**: 我们将预测模型直接用于异常检测，缺乏针对性设计。需要回归异常检测的本质：**识别偏离正常模式的模式**，而非**预测未来序列**。

---
*报告生成时间: 2025-05-29*
*模型版本: MTSMixer v1.0*
*数据集: machine-1, machine-2, machine-3* 