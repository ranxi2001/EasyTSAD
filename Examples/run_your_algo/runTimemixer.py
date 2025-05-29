from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import TSData

# 改进的TimeMixer组件
class ImprovedRevIN(nn.Module):
    """改进的可逆归一化层"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.std = torch.std(x, dim=1, keepdim=True) + self.eps
            x_norm = (x - self.mean) / self.std
            if self.affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            return x * self.std + self.mean
        return x

class EnhancedDecomposition(nn.Module):
    """增强的序列分解模块"""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # 使用多种分解方法
        # 1. 移动平均分解
        padding = self.kernel_size // 2
        x_padded = F.pad(x.transpose(1, 2), (padding, padding), mode='replicate')
        trend_ma = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1, padding=0)
        trend_ma = trend_ma.transpose(1, 2)
        
        # 2. 简单的频域分解
        x_freq = torch.fft.rfft(x, dim=1)
        frequencies = torch.fft.rfftfreq(seq_len, device=x.device)
        
        # 低频作为趋势（保留前20%的频率）
        cutoff = int(0.2 * len(frequencies))
        x_freq_trend = x_freq.clone()
        x_freq_trend[:, cutoff:] = 0
        trend_freq = torch.fft.irfft(x_freq_trend, n=seq_len, dim=1)
        
        # 结合两种趋势
        trend = 0.7 * trend_ma + 0.3 * trend_freq
        seasonal = x - trend
        
        return seasonal, trend

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 残差连接
        residual = x
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # 输出投影
        output = self.w_o(attn_output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output

class EnhancedMixingLayer(nn.Module):
    """增强的混合层"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)

class EnhancedTimeMixerModel(nn.Module):
    """增强的TimeMixer模型"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_features = configs.n_features
        
        # 组件
        self.revin = ImprovedRevIN(configs.n_features)
        self.input_projection = nn.Linear(configs.n_features, configs.d_model)
        self.decomposition = EnhancedDecomposition()
        
        # 注意力层
        self.attention = MultiHeadAttention(configs.d_model, configs.n_heads, configs.dropout)
        
        # 混合层
        self.seasonal_mixing = nn.ModuleList([
            EnhancedMixingLayer(configs.d_model, configs.d_ff, configs.dropout)
            for _ in range(configs.n_layers)
        ])
        self.trend_mixing = nn.ModuleList([
            EnhancedMixingLayer(configs.d_model, configs.d_ff, configs.dropout)
            for _ in range(configs.n_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.n_features)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model) * 0.02)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # 归一化
        x_norm = self.revin(x, mode='norm')
        
        # 投影到模型维度
        x_proj = self.input_projection(x_norm)
        
        # 添加位置编码
        x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        
        # 序列分解
        seasonal, trend = self.decomposition(x_proj)
        
        # 多层处理
        for seasonal_layer, trend_layer in zip(self.seasonal_mixing, self.trend_mixing):
            seasonal = seasonal_layer(seasonal)
            trend = trend_layer(trend)
        
        # 注意力机制
        seasonal = self.attention(seasonal) 
        trend = self.attention(trend)
        
        # 合并
        combined = seasonal + trend
        
        # 输出投影
        output = self.output_projection(combined)
        
        # 反归一化
        output = self.revin(output, mode='denorm')
        
        return output

class SimpleDataset(Dataset):
    """改进的数据集类"""
    def __init__(self, data, seq_len, stride=1, augment=False):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        
        if len(data) < seq_len:
            raise ValueError(f"Data length {len(data)} is less than sequence length {seq_len}")
            
        self.indices = list(range(0, len(data) - seq_len + 1, stride))
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        sample = torch.FloatTensor(self.data[start_idx:end_idx])
        
        # 数据增强
        if self.augment and torch.rand(1) > 0.5:
            # 添加轻微噪声
            noise = torch.randn_like(sample) * 0.01
            sample = sample + noise
            
        return sample

class TimeMixerConfig:
    """增强的TimeMixer配置类"""
    def __init__(self, **kwargs):
        # 默认参数
        self.seq_len = kwargs.get('seq_len', 96)
        self.d_model = kwargs.get('d_model', 128)  # 增加模型维度
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_layers = kwargs.get('n_layers', 3)  # 增加层数
        self.d_ff = kwargs.get('d_ff', 512)  # 增加前馈维度
        self.dropout = kwargs.get('dropout', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 10)  # 大幅增加训练轮数
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.patience = kwargs.get('patience', 10)
        self.n_features = None  # 将在训练时设置

if __name__ == "__main__":
    
    print("[LOG] 开始运行runTimeMixer.py")
    
    # Create a global controller
    gctrl = TSADController()
    print("[LOG] TSADController已创建")
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["machine-1", "machine-2", "machine-3"]
    dataset_types = "MTS"
    
    print("[LOG] 开始设置数据集")
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    """============= 实现TimeMixer算法 ============="""
    print("[LOG] 开始定义TimeMixer类")
    
    class TimeMixer(BaseMethod):
        def __init__(self, params: dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            
            # 配置
            self.config = TimeMixerConfig(**params)
            self.scaler = StandardScaler()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"[LOG] TimeMixer算法初始化完成，设备: {self.device}")
            
        def train_valid_phase(self, tsTrain: TSData):
            print(f"[LOG] TimeMixer开始训练，训练数据形状: {tsTrain.train.shape}")
            
            # 数据预处理
            train_data = tsTrain.train
            valid_data = tsTrain.valid
            
            # 设置特征数
            self.config.n_features = train_data.shape[1]
            
            # 标准化
            self.scaler.fit(train_data)
            train_data_scaled = self.scaler.transform(train_data)
            valid_data_scaled = self.scaler.transform(valid_data)
            
            # 创建数据集和数据加载器（使用数据增强）
            train_dataset = SimpleDataset(train_data_scaled, self.config.seq_len, augment=True)
            valid_dataset = SimpleDataset(valid_data_scaled, self.config.seq_len, augment=False)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                drop_last=True
            )
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=False,
                drop_last=False
            )
            
            # 创建模型
            self.model = EnhancedTimeMixerModel(self.config).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
            
            # 学习率调度器
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            
            # 训练
            best_valid_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.num_epochs):
                # 训练阶段
                self.model.train()
                train_losses = []
                
                for batch in train_loader:
                    batch = batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                    
                    # 添加正则化
                    l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                    loss = loss + 1e-6 * l2_reg
                    
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # 验证阶段
                self.model.eval()
                valid_losses = []
                
                with torch.no_grad():
                    for batch in valid_loader:
                        batch = batch.to(self.device)
                        output = self.model(batch)
                        loss = self.criterion(output, batch)
                        valid_losses.append(loss.item())
                
                train_loss = np.mean(train_losses)
                valid_loss = np.mean(valid_losses)
                
                # 更新学习率
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
                
                # 早停
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        print("Early stopping triggered")
                        break
            
            # 加载最佳模型
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state)
            
            print("[LOG] 训练完成")
            
        def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            print("[LOG] TimeMixer 不支持 all-in-one 模式")
            return
            
        def test_phase(self, tsData: TSData):
            print(f"[LOG] TimeMixer开始测试，测试数据形状: {tsData.test.shape}")
            
            test_data = tsData.test
            test_data_scaled = self.scaler.transform(test_data)
            
            # 创建测试数据集 - 使用滑动窗口，步长为1
            test_dataset = SimpleDataset(test_data_scaled, self.config.seq_len, stride=1)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            self.model.eval()
            reconstruction_errors = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    output = self.model(batch)
                    
                    # 计算多种异常分数
                    # 1. MSE误差
                    mse = torch.mean((batch - output) ** 2, dim=(1, 2))
                    
                    # 2. MAE误差
                    mae = torch.mean(torch.abs(batch - output), dim=(1, 2))
                    
                    # 3. 结合两种误差
                    combined_error = 0.7 * mse + 0.3 * mae
                    
                    reconstruction_errors.extend(combined_error.cpu().numpy())
            
            # 处理异常分数长度问题
            total_length = len(test_data)
            scores = np.zeros(total_length)
            
            # 前seq_len-1个点使用第一个窗口的分数
            if len(reconstruction_errors) > 0:
                scores[:self.config.seq_len-1] = reconstruction_errors[0]
                scores[self.config.seq_len-1:self.config.seq_len-1+len(reconstruction_errors)] = reconstruction_errors
            
            # 平滑处理
            from scipy import ndimage
            scores = ndimage.gaussian_filter1d(scores, sigma=1.0)
            
            # 标准化异常分数
            if len(scores) > 0 and np.std(scores) > 0:
                scores = (scores - np.mean(scores)) / np.std(scores)
                # 应用非线性变换增强异常信号
                scores = np.tanh(scores)
            
            self.__anomaly_score = scores
            print(f"[LOG] 异常分数计算完成，长度: {len(self.__anomaly_score)}")
            
        def anomaly_score(self) -> np.ndarray:
            print(f"[LOG] TimeMixer.anomaly_score() 调用")
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            if hasattr(self, 'model'):
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                param_info = f"增强TimeMixer模型\n"
                param_info += f"总参数量: {total_params}\n"
                param_info += f"配置:\n"
                for key, value in self.config.__dict__.items():
                    param_info += f"  {key}: {value}\n"
            else:
                param_info = "TimeMixer模型未初始化\n"
                
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    print("[LOG] TimeMixer类定义完成")
    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas
    training_schema = "naive"
    method = "TimeMixer"
    
    print(f"[LOG] 开始运行实验，method={method}, training_schema={training_schema}")
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "seq_len": 128,  # 增加序列长度
            "d_model": 128,  # 增加模型维度
            "n_heads": 8,
            "n_layers": 3,   # 增加层数
            "d_ff": 512,     # 增加前馈维度
            "dropout": 0.1,
            "batch_size": 32,
            "num_epochs": 10,  # 大幅增加训练轮数
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "patience": 10,
        },
        preprocess="z-score",
    )
    print("[LOG] 实验运行完成")
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    print("[LOG] 开始设置评估协议")
    gctrl.set_evals([
        PointF1PA(),
        EventF1PA(),
        EventF1PA(mode="squeeze")
    ])
    print("[LOG] 评估协议设置完成")

    print("[LOG] 开始执行评估")
    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
    print("[LOG] 评估执行完成")
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    print("[LOG] 开始绘图")
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
    print("[LOG] 绘图完成")
    
    print("[LOG] runTimeMixer.py执行完毕")
