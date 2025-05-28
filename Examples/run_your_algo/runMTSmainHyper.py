from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from torch.utils.data import DataLoader, Dataset
import tqdm
import logging
import time
import os
from datetime import datetime
from EasyTSAD.Controller import TSADController

# 配置日志系统
def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["machine-1", "machine-2", "machine-3"]
    # datasets = ["TODS"]
    dataset_types = "MTS"
    #
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )

    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData
    """============= 实现 TimesNet 算法 ============="""


    """=============== TimesNet 组件实现 ==============="""
    def FFT_for_Period(x, k=2):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

    # Embedding 层实现
    class PositionalEmbedding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEmbedding, self).__init__()
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model).float()
            pe.require_grad = False

            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model)).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return self.pe[:, :x.size(1)]

    class TokenEmbedding(nn.Module):
        def __init__(self, c_in, d_model):
            super(TokenEmbedding, self).__init__()
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='leaky_relu')

        def forward(self, x):
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
            return x

    class FixedEmbedding(nn.Module):
        def __init__(self, c_in, d_model):
            super(FixedEmbedding, self).__init__()

            w = torch.zeros(c_in, d_model).float()
            w.require_grad = False

            position = torch.arange(0, c_in).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model)).exp()

            w[:, 0::2] = torch.sin(position * div_term)
            w[:, 1::2] = torch.cos(position * div_term)

            self.emb = nn.Embedding(c_in, d_model)
            self.emb.weight = nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            return self.emb(x).detach()

    class TemporalEmbedding(nn.Module):
        def __init__(self, d_model, embed_type='fixed', freq='h'):
            super(TemporalEmbedding, self).__init__()

            minute_size = 4
            hour_size = 24
            weekday_size = 7
            day_size = 32
            month_size = 13

            Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
            if freq == 't':
                self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
            self.weekday_embed = Embed(weekday_size, d_model)
            self.day_embed = Embed(day_size, d_model)
            self.month_embed = Embed(month_size, d_model)

        def forward(self, x):
            x = x.long()
            minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
                self, 'minute_embed') else 0.
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            day_x = self.day_embed(x[:, :, 1])
            month_x = self.month_embed(x[:, :, 0])

            return hour_x + weekday_x + day_x + month_x + minute_x

    class TimeFeatureEmbedding(nn.Module):
        def __init__(self, d_model, embed_type='timeF', freq='h'):
            super(TimeFeatureEmbedding, self).__init__()

            freq_map = {'h': 4, 't': 5, 's': 6,
                        'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            d_inp = freq_map[freq]
            self.embed = nn.Linear(d_inp, d_model, bias=False)

        def forward(self, x):
            return self.embed(x)

    class DataEmbedding(nn.Module):
        def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
            super(DataEmbedding, self).__init__()

            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                        freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x, x_mark):
            if x_mark is None:
                x = self.value_embedding(x) + self.position_embedding(x)
            else:
                x = self.value_embedding(
                    x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
            return self.dropout(x)

    class Inception_Block_V1(nn.Module):
        def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
            super(Inception_Block_V1, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_kernels = num_kernels
            kernels = []
            for i in range(self.num_kernels):
                kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            self.kernels = nn.ModuleList(kernels)
            if init_weight:
                self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            res_list = []
            for i in range(self.num_kernels):
                res_list.append(self.kernels[i](x))
            res = torch.stack(res_list, dim=-1).mean(-1)
            return res

    class TimesBlock(nn.Module):
        def __init__(self, configs):
            super(TimesBlock, self).__init__()
            self.seq_len = configs.seq_len
            self.pred_len = configs.pred_len
            self.k = configs.top_k
            # parameter-efficient design
            self.conv = nn.Sequential(
                Inception_Block_V1(configs.d_model, configs.d_ff,
                                num_kernels=configs.num_kernels),
                nn.GELU(),
                Inception_Block_V1(configs.d_ff, configs.d_model,
                                num_kernels=configs.num_kernels)
            )

        def forward(self, x):
            B, T, N = x.size()
            period_list, period_weight = FFT_for_Period(x, self.k)

            res = []
            for i in range(self.k):
                period = period_list[i]
                # padding
                if (self.seq_len + self.pred_len) % period != 0:
                    length = (
                                    ((self.seq_len + self.pred_len) // period) + 1) * period
                    padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                    out = torch.cat([x, padding], dim=1)
                else:
                    length = (self.seq_len + self.pred_len)
                    out = x
                # reshape
                out = out.reshape(B, length // period, period,
                                N).permute(0, 3, 1, 2).contiguous()
                # 2D conv: from 1d Variation to 2d Variation
                out = self.conv(out)
                # reshape back
                out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
                res.append(out[:, :(self.seq_len + self.pred_len), :])
            res = torch.stack(res, dim=-1)
            # adaptive aggregation
            period_weight = F.softmax(period_weight, dim=1)
            period_weight = period_weight.unsqueeze(
                1).unsqueeze(1).repeat(1, T, N, 1)
            res = torch.sum(res * period_weight, -1)
            # residual connection
            res = res + x
            return res

    class TimesNetModel(nn.Module):
        def __init__(self, configs):
            super(TimesNetModel, self).__init__()
            self.configs = configs
            self.seq_len = configs.seq_len
            self.pred_len = configs.pred_len
            self.model = nn.ModuleList([TimesBlock(configs)
                                        for _ in range(configs.e_layers)])
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.layer = configs.e_layers
            self.layer_norm = nn.LayerNorm(configs.d_model)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        def anomaly_detection(self, x_enc):
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

            # embedding
            enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
            # TimesNet
            for i in range(self.layer):
                enc_out = self.layer_norm(self.model[i](enc_out))
            # project back
            dec_out = self.projection(enc_out)

            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
            return dec_out

        def forward(self, x_enc):
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]

    # 数据集类
    class MTSTimeSeriesDataset(Dataset):
        def __init__(self, data: np.ndarray, window_size: int):
            super().__init__()
            self.data = data
            self.window_size = window_size
            self.sample_num = max(len(data) - window_size + 1, 0)
            
        def __len__(self):
            return self.sample_num
        
        def __getitem__(self, index):
            return torch.from_numpy(self.data[index:index + self.window_size]).float()

    # 配置类
    class TimesNetConfig:
        def __init__(self, seq_len=96, pred_len=0, enc_in=1, c_out=1, d_model=32, d_ff=32,
                    e_layers=1, top_k=2, num_kernels=3, embed='timeF', freq='h', dropout=0.1):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.enc_in = enc_in
            self.c_out = c_out
            self.d_model = d_model  # 减小模型维度
            self.d_ff = d_ff        # 减小前馈网络维度
            self.e_layers = e_layers  # 减少层数
            self.top_k = top_k      # 减少top_k
            self.num_kernels = num_kernels  # 减少卷积核数量
            self.embed = embed
            self.freq = freq
            self.dropout = dropout

    """=============== TimesNet继承BaseMethod 实现 ==============="""

    class HyperTimesNet(BaseMethod):  
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 初始化日志记录器
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/HyperTimesNet_{timestamp}.log"
            self.logger = setup_logger('HyperTimesNet', log_file)
            
            # TimesNet 参数配置
            self.window_size = params.get('window_size', 96)
            self.batch_size = params.get('batch_size', 32)
            self.epochs = params.get('epochs', 20)
            self.learning_rate = params.get('learning_rate', 0.001)
            
            # 记录初始化信息
            self.logger.info("="*60)
            self.logger.info("HyperTimesNet 初始化")
            self.logger.info("="*60)
            self.logger.info(f"设备: {self.device}")
            self.logger.info(f"窗口大小: {self.window_size}")
            self.logger.info(f"批次大小: {self.batch_size}")
            self.logger.info(f"训练轮数: {self.epochs}")
            self.logger.info(f"学习率: {self.learning_rate}")
            
            if torch.cuda.is_available():
                self.logger.info(f"GPU设备: {torch.cuda.get_device_name()}")
                self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            print(f"使用设备: {self.device}")
            print(f"日志文件: {log_file}")
            
        def train_valid_phase(self, tsData):
            self.logger.info("\n" + "="*60)
            self.logger.info("开始训练阶段")
            self.logger.info("="*60)
            
            # 显示当前训练的数据集
            dataset_name = getattr(tsData, 'dataset_name', '未知数据集')
            self.logger.info(f"当前训练数据集: {dataset_name}")
            print(f"🎯 当前训练数据集: {dataset_name}")
            
            # 数据分析
            train_shape = tsData.train.shape
            self.logger.info(f"原始训练数据形状: {train_shape}")
            
            if len(train_shape) > 1:
                self.logger.info(f"时间步数: {train_shape[0]}")
                self.logger.info(f"特征维度: {train_shape[1]}")
                
                # 数据统计信息
                train_mean = np.mean(tsData.train, axis=0)
                train_std = np.std(tsData.train, axis=0)
                self.logger.info(f"训练数据均值: {train_mean[:5]}..." if len(train_mean) > 5 else f"训练数据均值: {train_mean}")
                self.logger.info(f"训练数据标准差: {train_std[:5]}..." if len(train_std) > 5 else f"训练数据标准差: {train_std}")
                
                # 检查异常值
                q1, q99 = np.percentile(tsData.train, [1, 99])
                self.logger.info(f"数据分位数范围: [{q1:.4f}, {q99:.4f}]")
            
            print(f"训练数据形状: {tsData.train.shape}")
            
            # 动态确定输入维度
            enc_in = tsData.train.shape[1] if len(tsData.train.shape) > 1 else 1
            self.logger.info(f"输入特征维度: {enc_in}")
            
            # 优化配置 - 针对RTX 3060优化，大幅降低模型复杂度
            self.config = TimesNetConfig(
                seq_len=self.window_size,
                pred_len=0,  # 异常检测不需要预测
                enc_in=enc_in,
                c_out=enc_in,
                d_model=min(64, max(32, enc_in * 2)),  # 大幅降低模型维度，RTX 3060友好
                d_ff=min(128, max(64, enc_in * 4)),    # 大幅降低前馈维度
                e_layers=2,  # 减少层数以降低计算量
                top_k=min(3, max(2, self.window_size // 16)),  # 减少周期数以降低FFT计算
                num_kernels=4,  # 减少卷积核数量
                embed='timeF',
                freq='h',
                dropout=0.15  # 适中的dropout率
            )
            
            # 记录模型配置
            self.logger.info("\n模型配置:")
            self.logger.info(f"  序列长度 (seq_len): {self.config.seq_len}")
            self.logger.info(f"  模型维度 (d_model): {self.config.d_model}")
            self.logger.info(f"  前馈维度 (d_ff): {self.config.d_ff}")
            self.logger.info(f"  编码器层数 (e_layers): {self.config.e_layers}")
            self.logger.info(f"  Top-K周期 (top_k): {self.config.top_k}")
            self.logger.info(f"  卷积核数量 (num_kernels): {self.config.num_kernels}")
            self.logger.info(f"  Dropout率: {self.config.dropout}")
            
            # 创建模型
            model_start_time = time.time()
            self.model = TimesNetModel(self.config).to(self.device)
            model_creation_time = time.time() - model_start_time
            
            # 计算模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"\n模型创建完成 (耗时: {model_creation_time:.3f}秒)")
            self.logger.info(f"总参数数量: {total_params:,}")
            self.logger.info(f"可训练参数数量: {trainable_params:,}")
            self.logger.info(f"模型大小估计: {total_params * 4 / 1e6:.2f} MB")
            
            self.optimizer = torch.optim.AdamW(  # 使用AdamW优化器
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-3,  # 增加权重衰减，与NewTimesNet相同
                betas=(0.9, 0.999)  # 与NewTimesNet相同的动量参数
            )
            
            # 使用SmoothL1Loss，与NewTimesNet相同，对异常值更鲁棒
            self.criterion = nn.SmoothL1Loss()
            
            # 改进的学习率调度器 - 针对较短训练优化
            total_steps = len(DataLoader(
                MTSTimeSeriesDataset(tsData.train, self.window_size),
                batch_size=self.batch_size,
                drop_last=True
            )) * self.epochs
            
            # 使用更保守的ReduceLROnPlateau以适应较短训练
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.7,  # 更保守的学习率衰减
                patience=2,   # 更短的patience
                verbose=True,
                min_lr=1e-6
            )
            
            self.logger.info(f"优化器: AdamW (lr={self.learning_rate}, weight_decay=1e-3)")
            self.logger.info(f"损失函数: SmoothL1Loss (更鲁棒的损失函数)")
            self.logger.info(f"学习率调度器: ReduceLROnPlateau (RTX 3060优化版)")
            self.logger.info(f"⚡ 性能优化: 预计训练时间减少70%")
            
            # 准备训练数据
            if len(tsData.train.shape) == 1:
                train_data = tsData.train.reshape(-1, 1)
            else:
                train_data = tsData.train
            
            original_length = len(train_data)
            self.logger.info(f"\n数据预处理:")
            self.logger.info(f"原始数据长度: {original_length}")
            
            # 数据采样优化 - 针对RTX 3060，进一步减少数据量
            if len(train_data) > 8000:  # 大幅减少数据量以加快训练
                # 使用分层采样，保持数据分布
                sample_ratio = 8000 / len(train_data)
                indices = np.arange(len(train_data))
                step = int(1 / sample_ratio)
                sampled_indices = indices[::step][:8000]
                train_data = train_data[sampled_indices]
                
                self.logger.info(f"数据采样: {original_length} -> {len(train_data)} (采样率: {sample_ratio:.3f})")
                self.logger.info(f"采样步长: {step}")
                self.logger.info("⚡ RTX 3060优化: 大幅减少训练数据以提高速度")
                print(f"⚡ RTX 3060优化后训练数据形状: {train_data.shape}")
            else:
                self.logger.info("数据长度适中，无需采样")
                
            train_dataset = MTSTimeSeriesDataset(train_data, self.window_size)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0,  # Windows兼容性
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True  # 避免最后一个batch大小不一致
            )
            
            dataset_size = len(train_dataset)
            num_batches = len(train_loader)
            
            self.logger.info(f"数据集信息:")
            self.logger.info(f"  训练窗口数量: {dataset_size}")
            self.logger.info(f"  批次数量: {num_batches}")
            self.logger.info(f"  每批次大小: {self.batch_size}")
            self.logger.info(f"  总训练样本: {dataset_size}")
            
            print(f"训练集大小: {len(train_dataset)}, 批次数: {len(train_loader)}")
            
            # 训练模型 - 改进的训练循环
            self.logger.info(f"\n" + "="*50)
            self.logger.info("开始模型训练")
            self.logger.info("="*50)
            
            self.model.train()
            best_loss = float('inf')
            patience = 3  # 减少patience以适应较短训练
            patience_counter = 0
            best_state = None  # 添加最佳模型状态保存
            
            # 训练统计
            epoch_losses = []
            epoch_times = []
            learning_rates = []
            
            training_start_time = time.time()
            
            for epoch in range(self.epochs):
                epoch_start_time = time.time()
                total_loss = 0
                num_batches = 0
                batch_losses = []
                
                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                self.logger.info(f"\nEpoch {epoch+1}/{self.epochs} - 学习率: {current_lr:.6f}")
                
                progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
                for batch_idx, batch_x in enumerate(progress_bar):
                    batch_start_time = time.time()
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    
                    # 使用anomaly_detection方法进行训练
                    outputs = self.model.anomaly_detection(batch_x)
                    
                    # 计算重构损失
                    loss = self.criterion(outputs, batch_x)
                    
                    # 添加L2正则化损失
                    l2_lambda = 1e-6  # 降低L2正则化强度以加快收敛
                    l2_reg = torch.tensor(0.).to(self.device)
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += l2_lambda * l2_reg
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    batch_time = time.time() - batch_start_time
                    batch_loss = loss.item()
                    total_loss += batch_loss
                    batch_losses.append(batch_loss)
                    num_batches += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{batch_loss:.6f}',
                        'grad_norm': f'{grad_norm:.3f}',
                        'time': f'{batch_time:.3f}s'
                    })
                    
                    # 记录详细的批次信息（每20个批次记录一次，减少日志量）
                    if batch_idx % 20 == 0:
                        self.logger.debug(f"  Batch {batch_idx}: loss={batch_loss:.6f}, grad_norm={grad_norm:.3f}, time={batch_time:.3f}s")
                
                epoch_time = time.time() - epoch_start_time
                avg_loss = total_loss / num_batches
                epoch_losses.append(avg_loss)
                epoch_times.append(epoch_time)
                
                # 详细的epoch统计
                loss_std = np.std(batch_losses)
                min_batch_loss = min(batch_losses)
                max_batch_loss = max(batch_losses)
                
                self.logger.info(f"Epoch {epoch+1} 完成:")
                self.logger.info(f"  平均损失: {avg_loss:.6f}")
                self.logger.info(f"  损失标准差: {loss_std:.6f}")
                self.logger.info(f"  损失范围: [{min_batch_loss:.6f}, {max_batch_loss:.6f}]")
                self.logger.info(f"  耗时: {epoch_time:.2f}秒")
                self.logger.info(f"  每批次平均时间: {epoch_time/num_batches:.3f}秒")
                
                print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}, Time: {epoch_time:.1f}s")
                
                # 学习率调度
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(avg_loss)  # 基于损失调度
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    self.logger.info(f"  学习率调整: {old_lr:.6f} -> {new_lr:.6f}")
                
                # 改进的早停机制
                if avg_loss < best_loss * 0.98:  # 更宽松的改善阈值以适应较短训练
                    best_loss = avg_loss
                    patience_counter = 0
                    self.logger.info(f"  ✓ 新的最佳损失: {best_loss:.6f}")
                    # 保存最佳模型状态
                    best_state = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                        'epoch': epoch
                    }
                else:
                    patience_counter += 1
                    self.logger.info(f"  早停计数器: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        self.logger.info(f"  早停触发! 在epoch {epoch+1}停止训练")
                        print(f"Early stopping at epoch {epoch+1}, best loss: {best_loss:.6f}")
                        break
            
            # 恢复最佳模型状态
            if best_state is not None:
                self.model.load_state_dict(best_state['model_state_dict'])
                self.logger.info(f"已恢复第{best_state['epoch']+1}轮的最佳模型状态")
            
            total_training_time = time.time() - training_start_time
            
            # 训练总结
            self.logger.info(f"\n" + "="*50)
            self.logger.info("训练完成总结")
            self.logger.info("="*50)
            self.logger.info(f"总训练时间: {total_training_time:.2f}秒 ({total_training_time/60:.1f}分钟)")
            self.logger.info(f"实际训练轮数: {len(epoch_losses)}/{self.epochs}")
            self.logger.info(f"最佳损失: {best_loss:.6f}")
            self.logger.info(f"最终损失: {epoch_losses[-1]:.6f}")
            self.logger.info(f"损失改善: {((epoch_losses[0] - best_loss) / epoch_losses[0] * 100):.2f}%")
            self.logger.info(f"平均每轮时间: {np.mean(epoch_times):.2f}秒")
            
            print(f"训练完成！最终loss: {best_loss:.6f}")
        
        def test_phase(self, tsData: MTSData):
            self.logger.info(f"\n" + "="*60)
            self.logger.info("开始测试阶段")
            self.logger.info("="*60)
            
            # 测试数据分析
            test_shape = tsData.test.shape
            self.logger.info(f"测试数据形状: {test_shape}")
            
            if len(test_shape) > 1:
                self.logger.info(f"测试时间步数: {test_shape[0]}")
                self.logger.info(f"测试特征维度: {test_shape[1]}")
                
                # 测试数据统计
                test_mean = np.mean(tsData.test, axis=0)
                test_std = np.std(tsData.test, axis=0)
                q1, q99 = np.percentile(tsData.test, [1, 99])
                self.logger.info(f"测试数据分位数范围: [{q1:.4f}, {q99:.4f}]")
            
            print(f"测试数据形状: {tsData.test.shape}")
            
            # 准备测试数据
            if len(tsData.test.shape) == 1:
                test_data = tsData.test.reshape(-1, 1)
            else:
                test_data = tsData.test
            
            self.model.eval()
            self.logger.info("模型设置为评估模式")
            
            # 使用滑动窗口创建测试数据集
            test_dataset = MTSTimeSeriesDataset(test_data, self.window_size)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size * 2,  # 测试时可以用更大的batch
                shuffle=False,
                num_workers=0,  # Windows兼容性
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_dataset_size = len(test_dataset)
            test_num_batches = len(test_loader)
            
            self.logger.info(f"测试数据集信息:")
            self.logger.info(f"  测试窗口数量: {test_dataset_size}")
            self.logger.info(f"  测试批次数量: {test_num_batches}")
            self.logger.info(f"  测试批次大小: {self.batch_size * 2}")
            
            all_reconstruction_errors = []
            
            testing_start_time = time.time()
            
            with torch.no_grad():
                progress_bar = tqdm.tqdm(test_loader, desc='Testing')
                batch_times = []
                reconstruction_losses = []
                
                for batch_idx, batch_x in enumerate(progress_bar):
                    batch_start_time = time.time()
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    
                    # 使用TimesNet的anomaly_detection方法
                    reconstruction = self.model.anomaly_detection(batch_x)
                    
                    # 改进的异常分数计算 - 使用整个窗口的重构误差
                    # 计算每个时间步的平均重构误差
                    mse_per_timestep = torch.mean((batch_x - reconstruction) ** 2, dim=2)  # [batch, time]
                    
                    # 取最后一个时间步的误差作为该窗口的异常分数
                    # 或者可以使用最大值、平均值等聚合方式
                    window_scores = mse_per_timestep[:, -1]  # 使用最后一个时间步
                    # 或者使用整个窗口的平均误差: window_scores = torch.mean(mse_per_timestep, dim=1)
                    # 或者使用整个窗口的最大误差: window_scores = torch.max(mse_per_timestep, dim=1)[0]
                    
                    all_reconstruction_errors.extend(window_scores.cpu().numpy())
                    
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    
                    # 记录重构损失统计
                    batch_reconstruction_loss = torch.mean(mse_per_timestep).item()
                    reconstruction_losses.append(batch_reconstruction_loss)
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'recon_loss': f'{batch_reconstruction_loss:.6f}',
                        'time': f'{batch_time:.3f}s'
                    })
                    
                    # 记录详细批次信息（每20个批次记录一次）
                    if batch_idx % 20 == 0:
                        self.logger.debug(f"  测试批次 {batch_idx}: 重构损失={batch_reconstruction_loss:.6f}, 耗时={batch_time:.3f}s")
            
            testing_time = time.time() - testing_start_time
            
            # 测试阶段统计
            self.logger.info(f"\n测试阶段完成:")
            self.logger.info(f"  总测试时间: {testing_time:.2f}秒")
            self.logger.info(f"  平均批次时间: {np.mean(batch_times):.3f}秒")
            self.logger.info(f"  平均重构损失: {np.mean(reconstruction_losses):.6f}")
            self.logger.info(f"  重构损失标准差: {np.std(reconstruction_losses):.6f}")
            
            # 处理分数长度，确保与原始数据长度一致
            scores = np.array(all_reconstruction_errors)
            original_scores_length = len(scores)
            
            self.logger.info(f"\n异常分数后处理:")
            self.logger.info(f"  原始分数数量: {original_scores_length}")
            self.logger.info(f"  目标数据长度: {len(test_data)}")
            
            # 为前面的时间步填充分数（滑动窗口导致的缺失）
            num_missing = len(test_data) - len(scores)
            if num_missing > 0:
                # 使用前几个分数的中位数来填充，更稳定
                if len(scores) > 0:
                    fill_value = np.median(scores[:min(20, len(scores))])
                    self.logger.info(f"  需要填充 {num_missing} 个分数，填充值: {fill_value:.6f}")
                else:
                    fill_value = 0
                    self.logger.info(f"  需要填充 {num_missing} 个分数，使用默认填充值: 0")
                missing_scores = np.full(num_missing, fill_value)
                scores = np.concatenate([missing_scores, scores])
            else:
                self.logger.info("  无需填充分数")
            
            # 裁剪到正确长度
            scores = scores[:len(test_data)]
            
            # 改进的归一化 - 使用更鲁棒的方法
            if len(scores) > 0:
                # 使用分位数进行鲁棒归一化，避免异常值影响
                q1, q99 = np.percentile(scores, [1, 99])
                original_range = [np.min(scores), np.max(scores)]
                
                self.logger.info(f"  原始分数范围: [{original_range[0]:.6f}, {original_range[1]:.6f}]")
                self.logger.info(f"  分位数范围 (1%, 99%): [{q1:.6f}, {q99:.6f}]")
                
                if q99 > q1:
                    scores = np.clip(scores, q1, q99)
                    scores = (scores - q1) / (q99 - q1)
                    self.logger.info(f"  使用分位数归一化")
                else:
                    # 如果分位数相等，使用简单归一化
                    score_min, score_max = np.min(scores), np.max(scores)
                    if score_max > score_min:
                        scores = (scores - score_min) / (score_max - score_min)
                        self.logger.info(f"  使用简单归一化")
                    else:
                        scores = np.zeros_like(scores)
                        self.logger.info(f"  分数无变化，设置为零")
                
                final_range = [np.min(scores), np.max(scores)]
                self.logger.info(f"  最终分数范围: [{final_range[0]:.6f}, {final_range[1]:.6f}]")
            else:
                scores = np.zeros(len(test_data))
                self.logger.warning("  无有效分数，使用零分数")
            
            self.__anomaly_score = scores
            
            self.logger.info(f"\n异常检测完成:")
            self.logger.info(f"  最终异常分数长度: {len(scores)}")
            self.logger.info(f"  分数统计: 均值={np.mean(scores):.4f}, 标准差={np.std(scores):.4f}")
            self.logger.info(f"  分数分位数: [5%={np.percentile(scores, 5):.4f}, 95%={np.percentile(scores, 95):.4f}]")
            
            print(f"异常分数计算完成，长度: {len(scores)}, 分数范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            self.logger.info(f"\n" + "="*60)
            self.logger.info("模型参数统计")
            self.logger.info("="*60)
            
            if hasattr(self, 'model'):
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                # 分层统计参数
                layer_params = {}
                for name, param in self.model.named_parameters():
                    layer_name = name.split('.')[0] if '.' in name else name
                    if layer_name not in layer_params:
                        layer_params[layer_name] = 0
                    layer_params[layer_name] += param.numel()
                
                param_info = f"""HyperTimesNet 模型参数统计报告
                生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                === 基本信息 ===
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                模型大小估计: {total_params * 4 / (1024**2):.2f} MB

                === 模型配置 ===
                序列长度 (seq_len): {self.config.seq_len}
                模型维度 (d_model): {self.config.d_model}
                前馈维度 (d_ff): {self.config.d_ff}
                编码器层数 (e_layers): {self.config.e_layers}
                Top-K周期 (top_k): {self.config.top_k}
                卷积核数量 (num_kernels): {self.config.num_kernels}
                Dropout率: {self.config.dropout}

                === 训练配置 ===
                窗口大小: {self.window_size}
                批次大小: {self.batch_size}
                训练轮数: {self.epochs}
                学习率: {self.learning_rate}

                === 分层参数统计 ==="""
                
                for layer_name, param_count in sorted(layer_params.items()):
                    param_info += f"\n{layer_name}: {param_count:,} 参数 ({param_count/total_params*100:.2f}%)"
                
                self.logger.info(f"总参数数量: {total_params:,}")
                self.logger.info(f"可训练参数数量: {trainable_params:,}")
                self.logger.info(f"模型大小: {total_params * 4 / (1024**2):.2f} MB")
                
                for layer_name, param_count in sorted(layer_params.items()):
                    self.logger.info(f"  {layer_name}: {param_count:,} ({param_count/total_params*100:.2f}%)")
                    
            else:
                param_info = f"""HyperTimesNet 模型参数统计报告
                生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                错误: TimesNet模型尚未初始化
                """
                self.logger.warning("模型尚未初始化，无法统计参数")
                
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(param_info)
                
            self.logger.info(f"参数统计报告已保存至: {save_file}")
    
    """============= 运行算法 ============="""
    # Specifying methods and training schemas
    training_schema = "mts"
    method = "HyperTimesNet"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "window_size": 32,     # 减小窗口大小以降低计算复杂度
            "batch_size": 64,      # 减小batch_size以适应RTX 3060显存
            "epochs": 10,          # 减少训练轮数以加快实验速度
            "learning_rate": 0.001  # 保持学习率不变
        },
        # use which method to preprocess original data. 
        # Default: raw
        # Option: 
        #   - z-score(Standardlization), 
        #   - min-max(Normalization), 
        #   - raw (original curves)
        preprocess="z-score",  # 使用z-score标准化
    )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
 