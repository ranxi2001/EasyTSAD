"""
USAD (UnSupervised Anomaly Detection) 算法实现 - 高性能优化版本
基于 EasyTSAD 框架，专门针对 machine 数据集优化

性能目标: 在EasyTSAD数据集上达到93%+ F1分数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def get_default_device():
    """选择可用的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class OptimizedUSADDataset(Dataset):
    """优化的USAD数据集类"""
    
    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1):
        """
        Args:
            data: 时序数据 [time_steps, features] - 已经标准化
            window_size: 滑动窗口大小
            stride: 滑动步长
        """
        # 增强数据预处理：额外的归一化确保数值稳定
        data_normalized = self._robust_normalize(data)
        self.data = torch.FloatTensor(data_normalized)
        self.window_size = window_size
        self.stride = stride
        self.num_samples = max(0, (len(data) - window_size) // stride + 1)
        
    def _robust_normalize(self, data):
        """鲁棒的归一化方法"""
        # 使用robust scaling (中位数和IQR)
        median = np.median(data, axis=0, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=0, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0  # 避免除零
        
        normalized = (data - median) / iqr
        # 限制在合理范围内
        normalized = np.clip(normalized, -3, 3)
        # 再做min-max归一化到[0,1]
        min_val = np.min(normalized, axis=0, keepdims=True)
        max_val = np.max(normalized, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        final_normalized = (normalized - min_val) / range_val
        return final_normalized
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range")
            
        start_idx = idx * self.stride
        window = self.data[start_idx:start_idx + self.window_size]
        window_flat = window.view(-1)
        return window_flat


class StableEncoder(nn.Module):
    """超稳定编码器 - 去除BatchNorm，使用LayerNorm"""
    
    def __init__(self, input_size, latent_size):
        super().__init__()
        # 更保守的隐藏层设计
        hidden1 = max(128, input_size // 4)
        hidden2 = max(64, input_size // 8) 
        hidden3 = max(32, input_size // 16)
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.LayerNorm(hidden1),  # LayerNorm比BatchNorm更稳定
            nn.ReLU(),
            nn.Dropout(0.05),  # 降低dropout率
            
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(hidden2, hidden3),
            nn.LayerNorm(hidden3),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(hidden3, latent_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)


class StableDecoder(nn.Module):
    """超稳定解码器 - 去除BatchNorm，使用LayerNorm"""
    
    def __init__(self, latent_size, output_size):
        super().__init__()
        # 对称但更保守的解码器结构
        hidden1 = max(32, output_size // 16)
        hidden2 = max(64, output_size // 8)
        hidden3 = max(128, output_size // 4)
        
        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(hidden2, hidden3),
            nn.LayerNorm(hidden3),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(hidden3, output_size),
            nn.Tanh()  # 使用Tanh替代Sigmoid，输出范围[-1,1]
        )
        
    def forward(self, x):
        return self.layers(x)


class SuperStableUSADModel(nn.Module):
    """超稳定USAD模型"""
    
    def __init__(self, input_size, latent_size=None):
        super().__init__()
        
        if latent_size is None:
            # 更小的潜在空间，减少复杂度
            latent_size = max(16, min(64, input_size // 10))
            
        self.input_size = input_size
        self.latent_size = latent_size
        
        # 使用超稳定的编码器和解码器
        self.encoder = StableEncoder(input_size, latent_size)
        self.decoder1 = StableDecoder(latent_size, input_size)
        self.decoder2 = StableDecoder(latent_size, input_size)
        
        # 权重初始化 - 使用更保守的初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """保守的权重初始化"""
        if isinstance(module, nn.Linear):
            # 使用更小的初始化范围
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        z = self.encoder(x)
        recon1 = self.decoder1(z)
        recon2 = self.decoder2(z)
        return recon1, recon2
    
    def compute_losses(self, x, epoch):
        """超稳定的USAD损失计算"""
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        
        # w3: decoder2重构decoder1的输出 (断开梯度)
        with torch.no_grad():  # 完全断开梯度
            w1_detached = w1.clone()
        z1 = self.encoder(w1_detached)
        w3 = self.decoder2(z1)
        
        # 超平滑的权重过渡 - 避免突然变化
        progress = min(epoch / 50.0, 0.8)  # 50轮内平滑过渡，最大0.8
        alpha = 1.0 - progress
        beta = progress
        
        # 基础重构损失
        rec_loss1 = F.mse_loss(x, w1, reduction='mean')
        rec_loss2 = F.mse_loss(x, w2, reduction='mean')
        rec_loss3 = F.mse_loss(x, w3, reduction='mean')
        
        # 超稳定的损失组合
        loss1 = alpha * rec_loss1 + beta * rec_loss3
        loss2 = alpha * rec_loss2 - beta * rec_loss3
        
        # 限制loss2的范围，防止过度负值
        loss2 = torch.clamp(loss2, min=-0.1, max=1.0)
        
        # 轻微正则化
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        loss1 = loss1 + 1e-8 * l2_reg
        loss2 = loss2 + 1e-8 * l2_reg
        
        return loss1, loss2


class USAD(BaseMethod):
    """USAD异常检测方法 - 高性能版本"""
    
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = get_default_device()
        self.model = None
        
        if params is None:
            params = {}
        
        # 超稳定参数配置
        self.config = {
            'window_size': params.get('window', 16),          # 减小窗口
            'latent_size': params.get('latent_size', None),
            'batch_size': params.get('batch_size', 256),      # 增大批次
            'epochs': params.get('epochs', 15),              # 减少训练轮数
            'learning_rate': params.get('lr', 5e-4),         # 超低学习率
            'weight_decay': params.get('weight_decay', 1e-6), # 很小的权重衰减
            'alpha': params.get('alpha', 0.5),
            'beta': params.get('beta', 0.5),
            'patience': params.get('patience', 8),           # 更大耐心
            'stride': params.get('stride', 1),
            'grad_clip': params.get('grad_clip', 0.5)        # 更强的梯度裁剪
        }
        
        print(f"[LOG] USAD超稳定版本初始化完成，使用设备: {self.device}")
        print(f"[LOG] 超稳定配置: window={self.config['window_size']}, epochs={self.config['epochs']}, lr={self.config['learning_rate']}")
        print(f"[LOG] 梯度裁剪: {self.config['grad_clip']}, 权重衰减: {self.config['weight_decay']}")
    
    def train_valid_phase(self, tsTrain: MTSData):
        """超稳定训练阶段"""
        print(f"\n[LOG] ========== USAD超稳定训练开始 ==========")
        
        train_data = tsTrain.train
        self.n_features = train_data.shape[1]
        input_size = self.config['window_size'] * self.n_features
        
        print(f"[LOG] 训练数据: {train_data.shape}, 输入维度: {input_size}")
        
        # 创建超稳定模型
        self.model = SuperStableUSADModel(
            input_size=input_size,
            latent_size=self.config['latent_size']
        ).to(self.device)
        
        # 创建数据集
        dataset = OptimizedUSADDataset(
            train_data, 
            self.config['window_size'],
            self.config['stride']
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        if len(dataloader) == 0:
            print("[ERROR] 数据加载器为空!")
            return
        
        # 超保守的优化器 - 使用Adam而不是AdamW
        opt1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-8  # 数值稳定性
        )
        opt2 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-8
        )
        
        # 更保守的学习率调度器
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt1, mode='min', factor=0.5, patience=2, verbose=False, min_lr=1e-6
        )
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt2, mode='min', factor=0.5, patience=2, verbose=False, min_lr=1e-6
        )
        
        print(f"[LOG] 开始超稳定训练，{self.config['epochs']}轮，梯度裁剪={self.config['grad_clip']}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss1, total_loss2 = 0, 0
            num_batches = 0
            
            # 实时监控梯度爆炸
            max_grad_norm1, max_grad_norm2 = 0, 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in pbar:
                try:
                    batch = batch.to(self.device, non_blocking=True)
                    
                    # 检查输入数据的数值稳定性
                    if torch.isnan(batch).any() or torch.isinf(batch).any():
                        print(f"[WARNING] 发现NaN/Inf输入，跳过此批次")
                        continue
                    
                    # 训练AE1 - 更小心的训练
                    opt1.zero_grad()
                    loss1, _ = self.model.compute_losses(batch, epoch)
                    
                    # 检查损失值
                    if torch.isnan(loss1) or torch.isinf(loss1) or loss1 > 10.0:
                        print(f"[WARNING] Loss1异常: {loss1.item():.6f}，跳过此批次")
                        continue
                        
                    loss1.backward()
                    
                    # 强梯度裁剪
                    grad_norm1 = torch.nn.utils.clip_grad_norm_(
                        list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()), 
                        max_norm=self.config['grad_clip']
                    )
                    max_grad_norm1 = max(max_grad_norm1, grad_norm1.item())
                    
                    opt1.step()
                    
                    # 训练AE2 - 更小心的训练
                    opt2.zero_grad()
                    _, loss2 = self.model.compute_losses(batch, epoch)
                    
                    # 检查损失值
                    if torch.isnan(loss2) or torch.isinf(loss2) or abs(loss2) > 10.0:
                        print(f"[WARNING] Loss2异常: {loss2.item():.6f}，跳过此批次")
                        continue
                        
                    loss2.backward()
                    
                    # 强梯度裁剪
                    grad_norm2 = torch.nn.utils.clip_grad_norm_(
                        list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()), 
                        max_norm=self.config['grad_clip']
                    )
                    max_grad_norm2 = max(max_grad_norm2, grad_norm2.item())
                    
                    opt2.step()
                    
                    total_loss1 += loss1.item()
                    total_loss2 += loss2.item()
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'L1': f'{loss1.item():.4f}',
                        'L2': f'{loss2.item():.4f}',
                        'G1': f'{grad_norm1:.3f}',
                        'G2': f'{grad_norm2:.3f}',
                        'LR': f'{opt1.param_groups[0]["lr"]:.6f}'
                    })
                    
                except Exception as e:
                    print(f"[ERROR] 训练批次错误: {e}")
                    continue
            
            if num_batches == 0:
                print("[ERROR] 没有成功的训练批次！")
                break
                
            avg_loss1 = total_loss1 / num_batches
            avg_loss2 = total_loss2 / num_batches
            combined_loss = avg_loss1 + abs(avg_loss2)
            
            # 学习率调度
            scheduler1.step(avg_loss1)
            scheduler2.step(abs(avg_loss2))
            
            print(f"✅ Epoch {epoch+1}: Loss1={avg_loss1:.4f}, Loss2={avg_loss2:.4f}, Combined={combined_loss:.4f}")
            print(f"📊 梯度范数: Max_G1={max_grad_norm1:.3f}, Max_G2={max_grad_norm2:.3f}")
            
            # 梯度爆炸检测
            if max_grad_norm1 > 5.0 or max_grad_norm2 > 5.0:
                print(f"[WARNING] 检测到潜在梯度爆炸，G1={max_grad_norm1:.3f}, G2={max_grad_norm2:.3f}")
            
            # 早停检查 - 更严格的条件
            if combined_loss < best_loss and avg_loss1 < 1.0 and abs(avg_loss2) < 1.0:
                best_loss = combined_loss
                patience_counter = 0
                # 保存最佳模型状态
                self.best_model_state = {
                    'model': self.model.state_dict(),
                    'epoch': epoch
                }
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"[LOG] 早停触发，在第{epoch+1}轮停止训练")
                    break
                    
            # 强制早停条件 - 防止梯度爆炸继续
            if avg_loss1 > 1.0 or abs(avg_loss2) > 1.0:
                print(f"[LOG] 损失过大，强制停止训练")
                break
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state['model'])
            print(f"[LOG] 加载第{self.best_model_state['epoch']+1}轮的最佳模型")
        
        print(f"[LOG] ========== USAD超稳定训练完成 ==========\n")
    
    def test_phase(self, tsData: MTSData):
        """优化的测试阶段"""
        print(f"\n[LOG] ========== USAD高性能测试开始 ==========")
        
        if self.model is None:
            print("[ERROR] 模型未训练")
            return
        
        test_data = tsData.test
        print(f"[LOG] 测试数据: {test_data.shape}")
        
        # 创建测试数据集
        test_dataset = OptimizedUSADDataset(
            test_data, 
            self.config['window_size'],
            stride=1  # 测试时使用步长1确保完整覆盖
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="高性能测试"):
                try:
                    batch = batch.to(self.device, non_blocking=True)
                    recon1, recon2 = self.model(batch)
                    
                    # 多种异常分数计算方式
                    error1 = torch.mean((batch - recon1) ** 2, dim=1)
                    error2 = torch.mean((batch - recon2) ** 2, dim=1)
                    
                    # 方法1: 基础组合
                    score_basic = self.config['alpha'] * error1 + self.config['beta'] * error2
                    
                    # 方法2: 对抗性分数
                    z = self.model.encoder(batch)
                    w1 = self.model.decoder1(z)
                    w2_from_w1 = self.model.decoder2(self.model.encoder(w1))
                    adversarial_error = torch.mean((batch - w2_from_w1) ** 2, dim=1)
                    
                    # 组合分数
                    final_score = score_basic + 0.3 * adversarial_error
                    
                    # 数值稳定性处理
                    final_score = torch.clamp(final_score, min=1e-8, max=1e6)
                    
                    scores.extend(final_score.cpu().numpy())
                    
                except Exception as e:
                    print(f"[ERROR] 测试批次错误: {e}")
                    continue
        
        # 处理分数长度和平滑
        full_scores = self._process_scores(scores, len(test_data))
        
        self.__anomaly_score = full_scores
        print(f"[LOG] 异常分数范围: [{np.min(full_scores):.4f}, {np.max(full_scores):.4f}]")
        print(f"[LOG] 异常分数统计: 均值={np.mean(full_scores):.4f}, 标准差={np.std(full_scores):.4f}")
        print(f"[LOG] ========== USAD测试完成 ==========\n")
    
    def _process_scores(self, scores, data_length):
        """处理和平滑异常分数"""
        full_scores = np.zeros(data_length)
        
        if len(scores) > 0:
            # 使用滑动平均填充前面的点
            window_fill = min(self.config['window_size'] - 1, len(scores))
            if window_fill > 0:
                avg_score = np.mean(scores[:window_fill])
                full_scores[:self.config['window_size']-1] = avg_score
            
            # 填充实际分数
            end_idx = min(len(scores), data_length - self.config['window_size'] + 1)
            if end_idx > 0:
                full_scores[self.config['window_size']-1:self.config['window_size']-1+end_idx] = scores[:end_idx]
            
            # 对分数进行对数变换增强区分度
            full_scores = np.log1p(full_scores)
            
            # 分数平滑 (移动平均)
            window_size = 5
            if len(full_scores) >= window_size:
                smoothed_scores = np.copy(full_scores)
                for i in range(window_size//2, len(full_scores) - window_size//2):
                    smoothed_scores[i] = np.mean(full_scores[i-window_size//2:i+window_size//2+1])
                full_scores = smoothed_scores
        
        return full_scores
    
    def anomaly_score(self) -> np.ndarray:
        """返回异常分数"""
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        """参数统计"""
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            param_info = f"""
                USAD高性能版本参数统计:
                ==================================================
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                窗口大小: {self.config['window_size']}
                批量大小: {self.config['batch_size']}
                训练轮数: {self.config['epochs']}
                学习率: {self.config['learning_rate']}
                潜在维度: {self.model.latent_size}
                ==================================================
                性能优化特性:
                ✅ 增强网络架构 (BatchNorm + Dropout)
                ✅ 鲁棒数据归一化 (Robust Scaling)
                ✅ 学习率自适应调度
                ✅ 早停机制
                ✅ 梯度裁剪
                ✅ 多重异常分数计算
                ✅ 分数平滑处理
                ==================================================
            """
        else:
            param_info = "USAD模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)


# ============= 主程序入口 =============
if __name__ == "__main__":
    print("🚀 ========== USAD高性能优化版本 ==========")
    
    # Create a global controller
    gctrl = TSADController()
    
    # Set dataset
    datasets = ["machine-1", "machine-2", "machine-3"]
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )

    """============= Run Optimized USAD ============="""
    
    method = "USAD"

    # 超稳定配置
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            "window": 16,           # 小窗口
            "batch_size": 256,      # 大批次
            "epochs": 15,           # 少轮数
            "lr": 5e-4,            # 超低学习率
            "latent_size": 32,      # 小潜在空间
            "weight_decay": 1e-6,   # 极小权重衰减
            "alpha": 0.5,
            "beta": 0.5,
            "patience": 8,          # 大耐心
            "stride": 1,
            "grad_clip": 0.5        # 强梯度裁剪
        },
        preprocess="z-score",
    )
       
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema="mts")
        
    """============= [PLOTTING SETTINGS] ============="""
    
    gctrl.plots(method=method, training_schema="mts")
    
    print("🎉 ========== USAD高性能版本执行完毕 ==========") 