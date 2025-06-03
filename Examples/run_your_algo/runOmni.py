"""
OmniAnomaly算法实现 - SMD数据集优化版本 (PyTorch实现)
基于EasyTSAD框架

性能目标: 在SMD数据集上达到95%+ F1分数
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


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
        
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return window


class Encoder(nn.Module):
    """编码器网络"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers=2):
        super().__init__()
        
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.hidden_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 均值和方差预测器
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        output, _ = self.rnn(x)
        hidden = output[:, -1, :]  # 取最后一个时间步
        
        hidden = self.hidden_fc(hidden)
        
        # 计算潜变量分布参数
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var


class Decoder(nn.Module):
    """解码器网络"""
    
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        
        self.latent_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim * 2)  # 输出均值和方差
        )
        
    def forward(self, z, seq_len):
        # z shape: [batch, latent_dim]
        hidden = self.latent_fc(z)
        
        # 扩展成序列
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        output, _ = self.rnn(hidden)
        output = self.output_fc(output)
        
        # 分离均值和方差
        mu, log_var = torch.chunk(output, 2, dim=-1)
        return mu, log_var


class OmniAnomalyModel(nn.Module):
    """OmniAnomaly模型"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers=2):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_layers)
        
    def reparameterize(self, mu, log_var):
        """重参数化采样"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # 编码
        mu, log_var = self.encoder(x)
        
        # 采样
        z = self.reparameterize(mu, log_var)
        
        # 解码
        recon_mu, recon_log_var = self.decoder(z, x.size(1))
        
        return recon_mu, recon_log_var, mu, log_var


class OmniAnomaly(BaseMethod):
    """OmniAnomaly异常检测方法 - SMD优化版本"""
    
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = get_default_device()
        self.model = None  # 初始化model属性为None
        
        if params is None:
            params = {}
            
        # SMD数据集优化配置
        self.config = {
            # 模型架构配置
            'input_dim': params.get('input_dim', 38),  # SMD默认38维
            'hidden_dim': params.get('hidden_dim', 500),
            'latent_dim': params.get('latent_dim', 8),
            'n_layers': params.get('n_layers', 2),
            'window_size': params.get('window_size', 100),
            
            # 训练参数
            'batch_size': params.get('batch_size', 128),
            'epochs': params.get('epochs', 10),
            'learning_rate': params.get('lr', 1e-3),
            'beta': params.get('beta', 0.01),  # KL损失权重
            
            # 评估参数
            'n_samples': params.get('n_samples', 10),  # 测试时的采样数
        }
        
        print(f"[LOG] OmniAnomaly SMD优化版本初始化完成")
        print(f"[LOG] 使用设备: {self.device}")
        print(f"[LOG] 配置: window={self.config['window_size']}, latent={self.config['latent_dim']}")
        
        # 创建模型
        self.model = OmniAnomalyModel(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            latent_dim=self.config['latent_dim'],
            n_layers=self.config['n_layers']
        ).to(self.device)
        
    def train_valid_phase(self, tsTrain: MTSData):
        """优化的训练阶段"""
        print(f"\n[LOG] ========== OmniAnomaly SMD优化训练开始 ==========")
        
        train_data = tsTrain.train
        self.n_features = train_data.shape[1]
        self.config['input_dim'] = self.n_features
        
        print(f"[LOG] 训练数据: {train_data.shape}")
        
        # 创建数据集和加载器
        dataset = TimeSeriesDataset(train_data, self.config['window_size'])
        train_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # 训练循环
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch in pbar:
                batch = batch.to(self.device)
                
                # 前向传播
                recon_mu, recon_log_var, mu, log_var = self.model(batch)
                
                # 简化的重构损失：使用MSE
                recon_loss = F.mse_loss(recon_mu, batch, reduction='sum')
                
                # KL散度: KL(q(z|x)||p(z)) where p(z) = N(0,I)
                kl_loss = 0.5 * torch.sum(
                    mu.pow(2) + log_var.exp() - log_var - 1
                )
                
                # 总损失 = 重构损失 + β * KL散度
                loss = recon_loss + self.config['beta'] * kl_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.2f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
            
            # 早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[LOG] 早停触发，在第{epoch+1}轮停止训练")
                    break
        
        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        print(f"[LOG] ========== OmniAnomaly SMD优化训练完成 ==========\n")
        
    def test_phase(self, tsData: MTSData):
        """优化的测试阶段"""
        print(f"\n[LOG] ========== OmniAnomaly SMD优化测试开始 ==========")
        
        test_data = tsData.test
        print(f"[LOG] 测试数据: {test_data.shape}")
        
        # 创建测试数据集
        dataset = TimeSeriesDataset(test_data, self.config['window_size'])
        test_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试中"):
                batch = batch.to(self.device)
                
                # 多次采样计算重构概率
                sample_scores = []
                for _ in range(self.config['n_samples']):
                    recon_mu, recon_log_var, _, _ = self.model(batch)
                    
                    # 计算重构误差作为异常分数
                    # 误差越大表示越异常
                    recon_error = F.mse_loss(recon_mu, batch, reduction='none')
                    # 对每个样本的每个时间步和特征求平均，然后取最后一个时间步
                    sample_scores.append(recon_error.mean(dim=-1)[:, -1].cpu().numpy())
                
                # 取平均
                batch_scores = np.mean(sample_scores, axis=0)
                scores.extend(batch_scores)
        
        # 处理分数
        full_scores = self._process_scores(scores, len(test_data))

        self.__anomaly_score = full_scores
        print(f"[LOG] 异常分数范围: [{np.min(full_scores):.4f}, {np.max(full_scores):.4f}]")
        print(f"[LOG] 异常分数统计: 均值={np.mean(full_scores):.4f}, 标准差={np.std(full_scores):.4f}")
        print(f"[LOG] ========== OmniAnomaly SMD优化测试完成 ==========\n")
        
    def _process_scores(self, scores, data_length):
        """处理异常分数"""
        # 填充开始的窗口
        full_scores = np.zeros(data_length)
        scores = np.array(scores)
        
        # 使用滑动平均填充前面的点
        window_fill = min(self.config['window_size'] - 1, len(scores))
        if window_fill > 0:
            full_scores[:self.config['window_size']-1] = np.mean(scores[:window_fill])
        
        # 填充实际分数
        full_scores[self.config['window_size']-1:self.config['window_size']-1+len(scores)] = scores
        
        # 移动平均平滑
        window = 5
        weights = np.ones(window) / window
        full_scores = np.convolve(full_scores, weights, mode='same')
        
        # 标准化到[0,1] (分数越高越异常)
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
            param_info = f"""
                OmniAnomaly SMD优化版本参数统计:
                ==================================================
                输入维度: {self.config['input_dim']}
                隐藏维度: {self.config['hidden_dim']}
                潜在维度: {self.config['latent_dim']}
                RNN层数: {self.config['n_layers']}
                窗口大小: {self.config['window_size']}
                批量大小: {self.config['batch_size']}
                训练轮数: {self.config['epochs']}
                学习率: {self.config['learning_rate']}
                ==================================================
                SMD优化特性:
                ✅ PyTorch深度学习框架
                ✅ GRU循环神经网络
                ✅ 变分自编码器
                ✅ 多重采样推断
                ✅ 梯度裁剪
                ✅ 早停机制
                ✅ 分数后处理优化
                ==================================================
            """
        else:
            param_info = "OmniAnomaly模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)


# ============= 主程序入口 =============
if __name__ == "__main__":
    print("🚀 ========== OmniAnomaly SMD优化版本 (PyTorch) ==========")
    
    # Create a global controller
    gctrl = TSADController()
    
    # Set dataset
    datasets = ["machine-1", "machine-2", "machine-3"]
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )

    """============= 运行优化的OmniAnomaly ============="""
    
    method = "OmniAnomaly"
    
    # SMD优化配置
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            'input_dim': 38,        # SMD默认38维
            'hidden_dim': 500,
            'latent_dim': 8,
            'n_layers': 2,
            'window_size': 100,
            'batch_size': 128,
            'epochs': 10,
            'lr': 1e-3,
            'beta': 0.01,           # KL损失权重
            'n_samples': 10         # 测试采样数
        },
        preprocess="z-score",
    )
       
    """============= 评估设置 ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema="mts")
        
    """============= 绘图设置 ============="""
    
    gctrl.plots(method=method, training_schema="mts")
    
    print("🎉 ========== OmniAnomaly SMD优化版本执行完毕 ==========") 