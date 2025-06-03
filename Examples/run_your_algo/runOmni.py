"""
OmniAnomaly算法实现 - 高性能优化版本 (PyTorch实现)
基于EasyTSAD框架

性能目标: 在SMD数据集上达到95%+ F1分数
主要优化: 修复损失函数、改进模型架构、优化GPU利用、增强训练稳定性
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
    """选择可用的设备并优化GPU设置"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 优化CUDA设置 for RTX 5080
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # 允许非确定性算法获得更好性能
        torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%显存
        print(f"[GPU] 使用设备: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] 显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    else:
        return torch.device('cpu')


class TimeSeriesDataset(Dataset):
    """优化的时间序列数据集"""
    
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
        
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return window


class AttentionEncoder(nn.Module):
    """增强的注意力编码器"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers=2, n_heads=8):
        super().__init__()
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1000, input_dim) * 0.1)
        
        # 双向GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向RNN
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 潜变量分布预测器
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 输入归一化和位置编码
        x = self.input_norm(x)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # 双向RNN
        rnn_out, _ = self.rnn(x)
        
        # 自注意力
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        
        # 残差连接
        rnn_out = rnn_out + attn_out
        
        # 全局最大池化和平均池化
        max_pool = torch.max(rnn_out, dim=1)[0]
        avg_pool = torch.mean(rnn_out, dim=1)
        
        # 特征融合
        features = max_pool + avg_pool
        features = self.feature_net(features)
        
        # 潜变量分布参数
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar


class AttentionDecoder(nn.Module):
    """增强的注意力解码器"""
    
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers=2, n_heads=8):
        super().__init__()
        
        # 潜变量投影
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 双向GRU
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影网络
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, z, seq_len):
        batch_size = z.size(0)
        
        # 潜变量投影
        hidden = self.latent_proj(z)
        
        # 扩展成序列
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 双向RNN
        rnn_out, _ = self.rnn(hidden)
        
        # 自注意力
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        
        # 残差连接
        output = rnn_out + attn_out
        
        # 输出投影
        output = self.output_net(output)
        
        return output


class AdvancedOmniAnomalyModel(nn.Module):
    """高级OmniAnomaly模型"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers=2, n_heads=8):
        super().__init__()
        
        self.encoder = AttentionEncoder(input_dim, hidden_dim, latent_dim, n_layers, n_heads)
        self.decoder = AttentionDecoder(latent_dim, hidden_dim, input_dim, n_layers, n_heads)
        
        # 模型参数
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """改进的重参数化采样"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # 测试时使用均值
        
    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon = self.decoder(z, x.size(1))
        
        return recon, mu, logvar


class OmniAnomaly(BaseMethod):
    """高性能OmniAnomaly异常检测方法"""
    
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = get_default_device()
        self.model = None
        
        if params is None:
            params = {}
            
        # 高性能配置for RTX 5080
        self.config = {
            # 模型架构配置 - 增强版
            'input_dim': params.get('input_dim', 38),
            'hidden_dim': params.get('hidden_dim', 512),  # 增大隐藏层
            'latent_dim': params.get('latent_dim', 16),   # 增大潜在维度
            'n_layers': params.get('n_layers', 3),        # 增加层数
            'n_heads': params.get('n_heads', 8),          # 多头注意力
            'window_size': params.get('window_size', 100),
            
            # 训练参数 - GPU优化
            'batch_size': params.get('batch_size', 256),   # 增大批量以充分利用GPU
            'epochs': params.get('epochs', 50),            # 增加训练轮数
            'learning_rate': params.get('lr', 1e-3),
            'beta': params.get('beta', 1.0),               # 增强KL约束
            'beta_annealing': params.get('beta_annealing', True),
            
            # 优化器配置
            'weight_decay': params.get('weight_decay', 1e-4),
            'warmup_epochs': params.get('warmup_epochs', 5),
            
            # 评估参数
            'n_samples': params.get('n_samples', 50),      # 增加采样数提高稳定性
            'score_window': params.get('score_window', 10), # 分数平滑窗口
        }
        
        print(f"[LOG] 🚀 高性能OmniAnomaly初始化完成")
        print(f"[LOG] 📊 模型配置: hidden={self.config['hidden_dim']}, latent={self.config['latent_dim']}, heads={self.config['n_heads']}")
        print(f"[LOG] 🎯 GPU优化: batch={self.config['batch_size']}, epochs={self.config['epochs']}")
        
    def train_valid_phase(self, tsTrain: MTSData):
        """高性能训练阶段"""
        print(f"\n[LOG] ========== 🚀 高性能训练开始 ==========")
        
        train_data = tsTrain.train
        self.n_features = train_data.shape[1]
        self.config['input_dim'] = self.n_features
        
        print(f"[LOG] 📊 训练数据: {train_data.shape}")
        
        # 创建高性能模型
        self.model = AdvancedOmniAnomalyModel(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            latent_dim=self.config['latent_dim'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads']
        ).to(self.device)
        
        # 混合精度训练for RTX 5080
        scaler = torch.cuda.amp.GradScaler()
        
        # 数据加载器 - 多进程优化
        dataset = TimeSeriesDataset(train_data, self.config['window_size'])
        train_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,  # 多进程加载
            pin_memory=True,  # 固定内存
            persistent_workers=True
        )
        
        # 优化器 - AdamW with warmup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.config['warmup_epochs']
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'] - self.config['warmup_epochs']
        )
        
        # 训练循环
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Beta退火策略
            if self.config['beta_annealing']:
                beta = min(self.config['beta'], epoch / 10.0)
            else:
                beta = self.config['beta']
            
            pbar = tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in pbar:
                batch = batch.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    recon, mu, logvar = self.model(batch)
                    
                    # 修复的损失函数 - 使用mean而不是sum
                    recon_loss = F.mse_loss(recon, batch, reduction='mean')
                    
                    # KL散度 - 正确的计算方式
                    kl_loss = -0.5 * torch.mean(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )
                    
                    # 确保KL损失为正
                    kl_loss = torch.clamp(kl_loss, min=0.0)
                    
                    # 总损失
                    loss = recon_loss + beta * kl_loss
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'beta': f'{beta:.3f}'
                })
            
            # 学习率调度
            if epoch < self.config['warmup_epochs']:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            avg_recon = total_recon_loss / len(train_loader)
            avg_kl = total_kl_loss / len(train_loader)
            
            print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[LOG] ⏰ 早停触发，在第{epoch+1}轮停止训练")
                    break
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        print(f"[LOG] ========== 🎉 高性能训练完成 ==========\n")
        
    def test_phase(self, tsData: MTSData):
        """高性能测试阶段"""
        print(f"\n[LOG] ========== 🔍 高性能测试开始 ==========")
        
        test_data = tsData.test
        print(f"[LOG] 📊 测试数据: {test_data.shape}")
        
        # 测试数据加载器
        dataset = TimeSeriesDataset(test_data, self.config['window_size'])
        test_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="🎯 测试中"):
                batch = batch.to(self.device, non_blocking=True)
                
                # 多次采样获得更稳定的分数
                batch_scores = []
                for _ in range(self.config['n_samples']):
                    with torch.cuda.amp.autocast():
                        recon, mu, logvar = self.model(batch)
                        
                        # 计算重构概率 - 负对数似然
                        recon_error = F.mse_loss(recon, batch, reduction='none')
                        # 对特征维度求平均，保留时间维度
                        point_scores = recon_error.mean(dim=-1)
                        # 取窗口最后一个点的分数
                        scores = point_scores[:, -1].cpu().numpy()
                        batch_scores.append(scores)
                
                # 取均值和标准差考虑不确定性
                mean_scores = np.mean(batch_scores, axis=0)
                std_scores = np.std(batch_scores, axis=0)
                # 结合均值和不确定性
                final_scores = mean_scores + 0.5 * std_scores
                all_scores.extend(final_scores)
        
        # 高级分数后处理
        full_scores = self._advanced_score_processing(all_scores, len(test_data))
        
        self.__anomaly_score = full_scores
        print(f"[LOG] 📈 异常分数范围: [{np.min(full_scores):.4f}, {np.max(full_scores):.4f}]")
        print(f"[LOG] 📊 异常分数统计: 均值={np.mean(full_scores):.4f}, 标准差={np.std(full_scores):.4f}")
        print(f"[LOG] ========== ✅ 高性能测试完成 ==========\n")
        
    def _advanced_score_processing(self, scores, data_length):
        """高级分数后处理"""
        scores = np.array(scores)
        full_scores = np.zeros(data_length)
        
        # 填充前面的窗口
        if len(scores) > 0:
            # 使用指数加权平均填充
            alpha = 0.3
            fill_value = scores[0]
            for i in range(self.config['window_size'] - 1):
                full_scores[i] = fill_value
                
        # 填充实际分数
        end_idx = min(self.config['window_size'] - 1 + len(scores), data_length)
        full_scores[self.config['window_size']-1:end_idx] = scores[:end_idx - self.config['window_size'] + 1]
        
        # 多重平滑处理
        # 1. 高斯平滑
        from scipy.ndimage import gaussian_filter1d
        full_scores = gaussian_filter1d(full_scores, sigma=2.0)
        
        # 2. 移动平均平滑
        window = self.config['score_window']
        weights = np.exp(np.linspace(-1, 0, window))  # 指数权重
        weights /= weights.sum()
        
        # 应用卷积平滑
        padded_scores = np.pad(full_scores, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded_scores, weights, mode='valid')
        full_scores = smoothed[:len(full_scores)]
        
        # 3. 异常值处理
        q75, q25 = np.percentile(full_scores, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        full_scores = np.clip(full_scores, None, upper_bound)
        
        # 4. 自适应归一化
        # 使用Robust Scaler
        median = np.median(full_scores)
        mad = np.median(np.abs(full_scores - median))
        if mad > 0:
            full_scores = (full_scores - median) / (1.4826 * mad)  # 1.4826是正态分布的修正因子
            full_scores = np.clip(full_scores, -3, 3)  # 限制在3倍MAD内
        
        # 5. 映射到[0,1]
        min_score = np.min(full_scores)
        max_score = np.max(full_scores)
        if max_score > min_score:
            full_scores = (full_scores - min_score) / (max_score - min_score)
        
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
                🚀 高性能OmniAnomaly参数统计:
                ==================================================
                📊 模型架构:
                - 输入维度: {self.config['input_dim']}
                - 隐藏维度: {self.config['hidden_dim']}
                - 潜在维度: {self.config['latent_dim']}
                - RNN层数: {self.config['n_layers']}
                - 注意力头数: {self.config['n_heads']}
                - 窗口大小: {self.config['window_size']}
                
                🎯 训练配置:
                - 批量大小: {self.config['batch_size']}
                - 训练轮数: {self.config['epochs']}
                - 学习率: {self.config['learning_rate']}
                - 权重衰减: {self.config['weight_decay']}
                - Beta权重: {self.config['beta']}
                
                💾 模型参数:
                - 总参数数: {total_params:,}
                - 可训练参数: {trainable_params:,}
                
                ==================================================
                🔥 高性能优化特性:
                ✅ RTX 5080 GPU加速
                ✅ 混合精度训练 (FP16)
                ✅ 多头自注意力机制
                ✅ 双向GRU + 残差连接
                ✅ AdamW优化器 + 余弦退火
                ✅ 学习率预热策略
                ✅ 梯度裁剪 + 早停
                ✅ 多重采样推断
                ✅ 高级分数后处理
                ✅ 鲁棒归一化
                ==================================================
            """
        else:
            param_info = "OmniAnomaly模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)


# ============= 主程序入口 =============
if __name__ == "__main__":
    print("🚀 ========== 高性能OmniAnomaly (RTX 5080优化版) ==========")
    
    # Create a global controller
    gctrl = TSADController()
    
    # Set dataset
    datasets = ["machine-1", "machine-2", "machine-3"]
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )

    """============= 运行高性能OmniAnomaly ============="""
    
    method = "OmniAnomaly"
    
    # RTX 5080优化配置
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            'input_dim': 38,         # SMD数据维度
            'hidden_dim': 512,       # 增大隐藏层
            'latent_dim': 16,        # 增大潜在维度
            'n_layers': 3,           # 增加层数
            'n_heads': 8,            # 多头注意力
            'window_size': 100,      # 时间窗口
            'batch_size': 256,       # GPU优化批量大小
            'epochs': 50,            # 充分训练
            'lr': 1e-3,              # 学习率
            'beta': 1.0,             # KL权重
            'beta_annealing': True,  # Beta退火
            'weight_decay': 1e-4,    # 权重衰减
            'warmup_epochs': 5,      # 预热轮数
            'n_samples': 50,         # 采样数
            'score_window': 10       # 分数平滑窗口
        },
        preprocess="z-score",
    )
       
    """============= 评估设置 ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema="mts")
        
    """============= 绘图设置 ============="""
    
    gctrl.plots(method=method, training_schema="mts")
    
    print("🎉 ========== 高性能OmniAnomaly执行完毕 ==========")