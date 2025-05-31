from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from sklearn.preprocessing import StandardScaler

from EasyTSAD.Controller import TSADController

print("[LOG] 🚀 开始初始化ReconstructionMTS算法...")

# ============================================================================
# 核心模块1: 多元时序编码器
# ============================================================================
class MultivariateTSEncoder(nn.Module):
    """
    多元时序编码器 - 捕获时间依赖和变量间关系
    """
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 时序卷积层 - 捕获局部时序模式
        self.temporal_conv1 = nn.Conv1d(input_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1)
        
        # 多变量注意力 - 捕获变量间关系
        self.multivar_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=max(1, hidden_dim//16), 
            batch_first=True
        )
        
        # 循环神经网络 - 捕获长期依赖（移除dropout避免警告）
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 压缩层
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # 时序卷积处理
        x_conv = F.relu(self.temporal_conv1(x.transpose(1,2)))  # [B, H//2, L]
        x_conv = F.relu(self.temporal_conv2(x_conv))  # [B, H, L]
        x_conv = x_conv.transpose(1,2)  # [B, L, H]
        
        # 多变量注意力
        x_attn, _ = self.multivar_attention(x_conv, x_conv, x_conv)  # [B, L, H]
        
        # 循环建模
        x_rnn, _ = self.rnn(x_attn)  # [B, L, H]
        
        # 压缩到潜在表征
        x_flat = x_rnn.reshape(B, -1)  # [B, L*H]
        z = self.compress(x_flat)  # [B, H]
        
        return z

# ============================================================================
# 核心模块2: 重构解码器
# ============================================================================
class ReconstructionDecoder(nn.Module):
    """
    重构解码器 - 从潜在表征恢复原始时序数据
    """
    def __init__(self, hidden_dim, output_dim, seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 解压缩层
        self.decompress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * seq_len)
        )
        
        # 循环解码器（移除dropout避免警告）
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 变量重构层
        self.var_reconstruct = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
        # 时序平滑层
        self.temporal_smooth = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        
    def forward(self, z):
        # z: [B, H]
        B = z.shape[0]
        
        # 解压缩
        x_decomp = self.decompress(z).reshape(B, self.seq_len, self.hidden_dim)  # [B, L, H]
        
        # 循环解码
        x_rnn, _ = self.rnn(x_decomp)  # [B, L, H]
        
        # 变量重构
        x_recon = self.var_reconstruct(x_rnn)  # [B, L, D]
        
        # 时序平滑
        x_smooth = self.temporal_smooth(x_recon.transpose(1,2)).transpose(1,2)  # [B, L, D]
        
        return x_smooth

# ============================================================================
# 核心模块3: 异常分数计算器
# ============================================================================
class AnomalyScoreComputer(nn.Module):
    """
    异常分数计算器 - 基于重构误差计算异常分数
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 重构误差加权网络
        self.error_weighter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # 时序异常检测器
        self.temporal_detector = nn.LSTM(1, 8, batch_first=True)
        self.temporal_scorer = nn.Linear(8, 1)
        
    def forward(self, original, reconstructed):
        # 计算重构误差
        recon_error = torch.abs(original - reconstructed)  # [B, L, D]
        
        # 加权重构误差
        error_weights = self.error_weighter(recon_error)  # [B, L, D]
        weighted_error = recon_error * error_weights  # [B, L, D]
        
        # 变量维度异常分数
        var_anomaly_score = torch.mean(weighted_error, dim=-1, keepdim=True)  # [B, L, 1]
        
        # 时序维度异常分数
        temp_features, _ = self.temporal_detector(var_anomaly_score)  # [B, L, 8]
        temp_anomaly_score = self.temporal_scorer(temp_features)  # [B, L, 1]
        
        # 综合异常分数
        final_score = torch.sigmoid(var_anomaly_score + temp_anomaly_score).squeeze(-1)  # [B, L]
        
        return final_score

# ============================================================================
# 核心模块4: 重构损失函数
# ============================================================================
class ReconstructionLoss(nn.Module):
    """
    重构损失函数 - 多目标损失优化
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.alpha = alpha  # 重构损失权重
        self.beta = beta    # 潜在表征正则化权重
        self.gamma = gamma  # 稀疏性正则化权重
    
    def forward(self, original, reconstructed, latent_repr):
        # 重构损失
        recon_loss = F.mse_loss(reconstructed, original)
        
        # 潜在表征正则化
        latent_reg = torch.mean(torch.norm(latent_repr, dim=1))
        
        # 稀疏性正则化
        sparsity_reg = torch.mean(torch.abs(latent_repr))
        
        total_loss = (self.alpha * recon_loss + 
                     self.beta * latent_reg + 
                     self.gamma * sparsity_reg)
        
        return total_loss, {
            'reconstruction': recon_loss,
            'latent_reg': latent_reg,
            'sparsity_reg': sparsity_reg
        }

# ============================================================================
# 核心模块5: 完整重构异常检测模型
# ============================================================================
class ReconstructionAnomalyDetector(nn.Module):
    """
    完整的重构基异常检测模型
    """
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        # 编码器
        self.encoder = MultivariateTSEncoder(input_dim, hidden_dim, seq_len)
        # 解码器
        self.decoder = ReconstructionDecoder(hidden_dim, input_dim, seq_len)
        # 异常分数计算器
        self.anomaly_computer = AnomalyScoreComputer(input_dim)
    
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # 编码
        latent_repr = self.encoder(x)  # [B, H]
        
        # 解码重构
        reconstructed = self.decoder(latent_repr)  # [B, L, D]
        
        # 计算异常分数
        anomaly_scores = self.anomaly_computer(x, reconstructed)  # [B, L]
        
        return {
            'reconstructed': reconstructed,
            'anomaly_scores': anomaly_scores,
            'latent_repr': latent_repr
        }

# ============================================================================
# 智能后处理器
# ============================================================================
class IntelligentPostProcessor:
    """
    智能后处理器 - 优化异常分数
    """
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.threshold = None
        self.fitted = False
    
    def fit(self, train_scores):
        """在训练分数上拟合阈值"""
        self.threshold = np.percentile(train_scores, (1 - self.contamination) * 100)
        self.fitted = True
    
    def process(self, raw_scores):
        """处理原始异常分数"""
        if not self.fitted:
            # 如果未拟合，使用自适应阈值
            self.threshold = np.percentile(raw_scores, 95)
        
        # 平滑处理
        smoothed_scores = self._smooth_scores(raw_scores)
        
        # 标准化
        normalized_scores = self._normalize_scores(smoothed_scores)
        
        return normalized_scores
    
    def _smooth_scores(self, scores, window=5):
        """平滑异常分数"""
        if len(scores) < window:
            return scores
        
        smoothed = np.convolve(scores, np.ones(window)/window, mode='same')
        return smoothed
    
    def _normalize_scores(self, scores):
        """标准化异常分数"""
        scores = np.array(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score > 0:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = scores
        
        return normalized

print("[LOG] 🔧 核心模块定义完成")

# ============================================================================
# 主算法类: ReconstructionMTS
# ============================================================================
if __name__ == "__main__":
    # 创建控制器
    gctrl = TSADController()
    print("[LOG] TSADController已创建")
        
    """============= [DATASET SETTINGS] ============="""
    # 指定数据集
    datasets = ["machine-1", "machine-2", "machine-3"]
    dataset_types = "MTS"
    
    print("[LOG] 开始设置数据集")
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="MTS",
        # dirname="../../datasets",
        dirname="./datasets", # 项目根目录中的相对路径 就是当前路径
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    """============= Implement your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData
    
    print("[LOG] 开始定义ReconstructionMTS类")

    class ReconstructionMTS(BaseMethod):
        def __init__(self, params: dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.criterion = None
            self.post_processor = IntelligentPostProcessor()
            self.window_size = 64
            self.training_scores = []
            self.scaler = StandardScaler()
            
            # 模型参数
            self.hidden_dim = params.get('hidden_dim', 64)
            self.learning_rate = params.get('learning_rate', 0.001)
            self.epochs = params.get('epochs', 30)
            self.batch_size = params.get('batch_size', 16)
            
            print(f"[LOG] 🤖 ReconstructionMTS.__init__() 调用，使用设备: {self.device}")
            print(f"[LOG] 📊 参数配置 - hidden_dim: {self.hidden_dim}, epochs: {self.epochs}")
            
        def _build_model(self, input_dim, seq_len=64):
            """构建重构异常检测模型"""
            print(f"[LOG] 🔧 构建ReconstructionMTS模型，输入维度: {input_dim}, 序列长度: {seq_len}")
            
            self.window_size = seq_len
            
            self.model = ReconstructionAnomalyDetector(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                seq_len=seq_len
            ).to(self.device)
            
            self.criterion = ReconstructionLoss(alpha=1.0, beta=0.1, gamma=0.01)
            
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"[LOG] ✅ ReconstructionMTS模型构建成功，参数数量: {param_count}")
            
        def _create_windows(self, data, window_size, stride=1):
            """创建滑动窗口"""
            windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
            return torch.stack(windows) if windows else torch.unsqueeze(data[:window_size], 0)
        
        def train_valid_phase(self, tsData):
            print(f"[LOG] 🎓 ReconstructionMTS.train_valid_phase() 调用，数据形状: {tsData.train.shape}")
            
            # 数据预处理
            train_data_scaled = self.scaler.fit_transform(tsData.train)
            train_data = torch.FloatTensor(train_data_scaled).to(self.device)
            seq_len, input_dim = train_data.shape
            
            window_size = min(seq_len, 64)
            stride = max(1, window_size // 8)
            
            self._build_model(input_dim, window_size)
            
            # 创建训练窗口
            train_windows = self._create_windows(train_data, window_size, stride)
            print(f"[LOG] 📊 训练窗口数量: {train_windows.shape[0]}")
            
            # 训练模型
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
            
            batch_size = min(self.batch_size, train_windows.shape[0])
            
            best_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            print(f"[LOG] 🚀 开始重构训练，epochs: {self.epochs}, batch_size: {batch_size}")
            
            for epoch in range(self.epochs):
                total_loss = 0
                total_losses = {'reconstruction': 0, 'latent_reg': 0, 'sparsity_reg': 0}
                num_batches = 0
                
                indices = torch.randperm(train_windows.shape[0])
                
                for i in range(0, train_windows.shape[0], batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch = train_windows[batch_indices]
                    
                    optimizer.zero_grad()
                    
                    try:
                        # 前向传播
                        outputs = self.model(batch)
                        
                        # 计算损失
                        loss, loss_dict = self.criterion(
                            batch, 
                            outputs['reconstructed'], 
                            outputs['latent_repr']
                        )
                        
                        # 反向传播
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        for key, value in loss_dict.items():
                            if key in total_losses:
                                total_losses[key] += value.item()
                        num_batches += 1
                        
                        # 收集训练异常分数用于后处理器
                        if epoch == self.epochs - 1:  # 最后一轮收集
                            with torch.no_grad():
                                scores = outputs['anomaly_scores'].cpu().numpy()
                                self.training_scores.extend(scores.flatten())
                        
                    except Exception as e:
                        print(f"[WARNING] 训练批次失败: {e}")
                        continue
                
                if num_batches == 0:
                    print("[ERROR] 所有训练批次都失败")
                    break
                    
                avg_loss = total_loss / num_batches
                scheduler.step()
                
                # 早停机制
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"[LOG] ⏰ 早停在第 {epoch+1} 轮，最佳损失: {best_loss:.6f}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"[LOG] 📈 Epoch {epoch+1}, Loss: {avg_loss:.6f}, "
                          f"Recon: {total_losses['reconstruction']/num_batches:.6f}")
            
            # 训练后处理器
            if self.training_scores:
                self.post_processor.fit(np.array(self.training_scores))
                print(f"[LOG] 🧠 后处理器训练完成，使用 {len(self.training_scores)} 个样本")
            
            print("[LOG] ✅ ReconstructionMTS训练完成")
        
        def test_phase(self, tsData: MTSData):
            print(f"[LOG] 🔍 ReconstructionMTS.test_phase() 调用，测试数据形状: {tsData.test.shape}")
            
            if self.model is None:
                print("[ERROR] 模型未训练")
                return
            
            # 数据预处理
            test_data_scaled = self.scaler.transform(tsData.test)
            test_data = torch.FloatTensor(test_data_scaled).to(self.device)
            seq_len, input_dim = test_data.shape
            
            self.model.eval()
            scores = []
            
            print(f"[LOG] 🎯 开始重构异常检测，序列长度: {seq_len}")
            
            with torch.no_grad():
                for i in range(seq_len):
                    start_idx = max(0, i - self.window_size + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < self.window_size:
                        # 对于序列开头，使用零填充
                        window = torch.zeros(self.window_size, input_dim).to(self.device)
                        actual_data = test_data[start_idx:end_idx, :]
                        window[-actual_data.shape[0]:, :] = actual_data
                    else:
                        window = test_data[start_idx:end_idx, :]
                    
                    window_batch = window.unsqueeze(0)  # [1, window_size, input_dim]
                    
                    try:
                        outputs = self.model(window_batch)
                        
                        # 获取当前时间点的异常分数
                        current_score = outputs['anomaly_scores'][0, -1].item()  # 最后一个时间点
                        scores.append(current_score)
                        
                    except Exception as e:
                        if i < 10:
                            print(f"[WARNING] 窗口 {i} 预测失败: {e}")
                        scores.append(0.0)
            
            scores = np.array(scores)
            
            # 后处理
            processed_scores = self.post_processor.process(scores)
            
            self.__anomaly_score = processed_scores
            print(f"[LOG] ✅ 异常检测完成，分数范围: [{np.min(processed_scores):.4f}, {np.max(processed_scores):.4f}]")
        
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            if self.model is not None:
                param_count = sum(p.numel() for p in self.model.parameters())
                param_info = f"""🚀 ReconstructionMTS (重构基异常检测器) 模型信息:
                
                📋 模型类型: ReconstructionMTS - 基于重构的多元时序异常检测
                💻 使用设备: {self.device}
                🔢 模型参数数量: {param_count}
                
                🏗️ 核心架构:
                ✅ 多元时序编码器 (Multivariate TS Encoder)
                ✅ 重构解码器 (Reconstruction Decoder)  
                ✅ 异常分数计算器 (Anomaly Score Computer)
                ✅ 智能后处理器 (Intelligent Post-Processor)
                
                🚀 技术创新:
                1. 编码器-解码器架构 - 学习正常数据内在结构
                2. 多维重构误差 - 时间+变量维度异常检测
                3. 混合损失优化 - 重构+正则化+稀疏性约束
                4. 智能后处理 - 分数平滑和自适应标准化
                5. 无监督学习 - 仅需正常数据训练
                
                🎯 设计目标: 实现高精度重构基异常检测
                📈 预期效果: Point F1 85%+, Event F1 70%+
                🔧 工程优势: 结构简洁，泛化能力强
                """
                with open(save_file, 'w', encoding='utf-8') as f:
                    f.write(param_info)
                print(f"[LOG] 📊 参数统计已保存到 {save_file}")

    print("[LOG] ✅ ReconstructionMTS类定义完成")
    
    """============= Run your algo. ============="""
    training_schema = "mts"
    method = "ReconstructionMTS"
    
    print(f"[LOG] 🚀 开始运行ReconstructionMTS实验，method={method}, training_schema={training_schema}")
    
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        preprocess="z-score",
    )
    print("[LOG] 🎉 ReconstructionMTS实验运行完成")
       
    """============= [EVALUATION SETTINGS] ============="""
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    print("[LOG] 开始设置评估协议")
    gctrl.set_evals([
        PointF1PA(),
        EventF1PA(),
        EventF1PA(mode="squeeze")
    ])
    print("[LOG] 评估协议设置完成")

    print("[LOG] 🔍 开始执行ReconstructionMTS评估")
    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
    print("[LOG] ✅ ReconstructionMTS评估执行完成")
        
    """============= [PLOTTING SETTINGS] ============="""
    print("[LOG] 📊 开始ReconstructionMTS结果绘图")
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
    print("[LOG] 🎨 ReconstructionMTS绘图完成")
    
    print("[LOG] 🏆 ReconstructionMTS (重构基异常检测器) 执行完毕")
    print("[LOG] 🎯 期待优秀的重构基异常检测性能！") 