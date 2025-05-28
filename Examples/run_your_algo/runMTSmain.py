from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from torch.utils.data import DataLoader, Dataset
import tqdm
from EasyTSAD.Controller import TSADController


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
        def __init__(self, seq_len=96, pred_len=0, enc_in=1, c_out=1, d_model=64, d_ff=128,
                    e_layers=2, top_k=3, num_kernels=4, embed='timeF', freq='h', dropout=0.2):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.enc_in = enc_in
            self.c_out = c_out
            self.d_model = d_model  # 增加模型维度
            self.d_ff = d_ff        # 增加前馈网络维度
            self.e_layers = e_layers  # 增加层数
            self.top_k = top_k      # 增加top_k以捕获更多周期性
            self.num_kernels = num_kernels  # 增加卷积核数量
            self.embed = embed
            self.freq = freq
            self.dropout = dropout   # 增加dropout防止过拟合

    """=============== TimesNet继承BaseMethod 实现 ==============="""

    class NewTimesNet(BaseMethod):  
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # TimesNet 参数配置
            self.window_size = params.get('window_size', 48)  # 增大窗口大小
            self.batch_size = params.get('batch_size', 128)   # 增大batch_size
            self.epochs = params.get('epochs', 20)            # 增加训练轮数
            self.learning_rate = params.get('learning_rate', 0.001)
            
            print(f"使用设备: {self.device}")
            
        def train_valid_phase(self, tsData):
            print(f"训练数据形状: {tsData.train.shape}")
            
            # 动态确定输入维度
            enc_in = tsData.train.shape[1] if len(tsData.train.shape) > 1 else 1
            
            # 优化配置 - 专门针对异常检测调整
            self.config = TimesNetConfig(
                seq_len=self.window_size,
                pred_len=0,  # 异常检测不需要预测
                enc_in=enc_in,
                c_out=enc_in,
                d_model=min(128, max(64, enc_in * 4)),  # 增加模型容量
                d_ff=min(256, max(128, enc_in * 8)),    # 增加前馈网络容量
                e_layers=3,  # 增加层数
                top_k=min(4, max(3, self.window_size // 12)),  # 增加周期性捕获
                num_kernels=6,  # 增加卷积核数量
                embed='timeF',
                freq='h',
                dropout=0.2  # 增加dropout
            )
            
            # 创建模型
            self.model = TimesNetModel(self.config).to(self.device)
            
            # 使用AdamW优化器，增加权重衰减
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-3,  # 增加权重衰减
                betas=(0.9, 0.999)  # 调整动量参数
            )
            
            # 使用SmoothL1Loss，对异常值更鲁棒
            self.criterion = nn.SmoothL1Loss()
            
            # 改进的学习率调度器
            self.scheduler = torch.optim.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                epochs=self.epochs,
                steps_per_epoch=len(DataLoader(
                    MTSTimeSeriesDataset(tsData.train, self.window_size),
                    batch_size=self.batch_size
                )),
                pct_start=0.3,  # 前30%时间用于预热
                div_factor=10.0,
                final_div_factor=1e4
            )
            
            # 准备训练数据
            if len(tsData.train.shape) == 1:
                train_data = tsData.train.reshape(-1, 1)
            else:
                train_data = tsData.train
            
            # 数据采样优化 - 更智能的采样策略
            if len(train_data) > 15000:
                # 使用分层采样，保持数据分布
                sample_ratio = 15000 / len(train_data)
                indices = np.arange(len(train_data))
                step = int(1 / sample_ratio)
                sampled_indices = indices[::step][:15000]
                train_data = train_data[sampled_indices]
                print(f"智能采样后训练数据形状: {train_data.shape}")
                
            train_dataset = MTSTimeSeriesDataset(train_data, self.window_size)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0,  # Windows兼容性
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True  # 避免最后一个batch大小不一致
            )
            
            print(f"训练集大小: {len(train_dataset)}, 批次数: {len(train_loader)}")
            
            # 训练模型 - 改进的训练循环
            self.model.train()
            best_loss = float('inf')
            patience = 5  # 增加patience
            patience_counter = 0
            best_state = None
            
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = 0
                
                progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
                for batch_x in progress_bar:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    
                    # 使用anomaly_detection方法进行训练
                    outputs = self.model.anomaly_detection(batch_x)
                    
                    # 计算重构损失
                    loss = self.criterion(outputs, batch_x)
                    
                    # 添加正则化损失
                    l2_lambda = 1e-5
                    l2_reg = torch.tensor(0.).to(self.device)
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += l2_lambda * l2_reg
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
                
                # 改进的早停机制
                if avg_loss < best_loss * 0.99:  # 需要有更明显的改进
                    best_loss = avg_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_state = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                        'epoch': epoch
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}, best loss: {best_loss:.6f}")
                        break
            
            # 恢复最佳模型状态
            if best_state is not None:
                self.model.load_state_dict(best_state['model_state_dict'])
            
            print(f"训练完成！最终loss: {best_loss:.6f}")
            
        def test_phase(self, tsData: MTSData):
            print(f"测试数据形状: {tsData.test.shape}")
            
            # 准备测试数据
            if len(tsData.test.shape) == 1:
                test_data = tsData.test.reshape(-1, 1)
            else:
                test_data = tsData.test
            
            self.model.eval()
            
            # 使用滑动窗口创建测试数据集
            test_dataset = MTSTimeSeriesDataset(test_data, self.window_size)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size * 2,  # 测试时可以用更大的batch
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            all_reconstruction_errors = []
            
            with torch.no_grad():
                progress_bar = tqdm.tqdm(test_loader, desc='Testing')
                for batch_x in progress_bar:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    
                    # 使用anomaly_detection方法
                    reconstruction = self.model.anomaly_detection(batch_x)
                    
                    # 改进的异常分数计算
                    # 1. 计算每个时间步的重构误差
                    errors = (batch_x - reconstruction) ** 2
                    
                    # 2. 计算每个特征的权重（基于方差）
                    feature_weights = 1.0 / (torch.var(batch_x, dim=1, keepdim=True) + 1e-5)
                    feature_weights = feature_weights / feature_weights.sum(dim=2, keepdim=True)
                    
                    # 3. 加权平均重构误差
                    weighted_errors = errors * feature_weights
                    mse_per_timestep = torch.sum(weighted_errors, dim=2)  # [batch, time]
                    
                    # 4. 使用指数加权移动平均计算最终分数
                    alpha = 0.3
                    window_scores = torch.zeros(mse_per_timestep.shape[0], device=self.device)
                    for t in range(mse_per_timestep.shape[1]):
                        window_scores = alpha * mse_per_timestep[:, t] + (1 - alpha) * window_scores
                    
                    all_reconstruction_errors.extend(window_scores.cpu().numpy())
            
            # 处理分数长度
            scores = np.array(all_reconstruction_errors)
            
            # 填充缺失的时间步
            num_missing = len(test_data) - len(scores)
            if num_missing > 0:
                if len(scores) > 0:
                    # 使用前20个分数的中位数，更稳定
                    fill_value = np.median(scores[:min(20, len(scores))])
                else:
                    fill_value = 0
                missing_scores = np.full(num_missing, fill_value)
                scores = np.concatenate([missing_scores, scores])
            
            # 裁剪到正确长度
            scores = scores[:len(test_data)]
            
            # 改进的分数后处理
            if len(scores) > 0:
                # 1. 移除极端异常值
                q1, q99 = np.percentile(scores, [1, 99])
                scores = np.clip(scores, q1, q99)
                
                # 2. 应用指数变换突出异常
                scores = np.exp(scores / scores.std()) - 1
                
                # 3. 最终归一化
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                
                # 4. 平滑处理
                window_size = 5
                kernel = np.ones(window_size) / window_size
                scores = np.convolve(scores, kernel, mode='same')
            
            self.__anomaly_score = scores
            print(f"异常分数计算完成，长度: {len(scores)}, 分数范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            if hasattr(self, 'model'):
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                param_info = f"""
                TimesNet模型参数统计:
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                模型配置:
                - seq_len: {self.config.seq_len}
                - d_model: {self.config.d_model}
                - e_layers: {self.config.e_layers}
                - top_k: {self.config.top_k}
                """
            else:
                param_info = "TimesNet模型尚未初始化"
                
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    """============= 运行算法 ============="""
    # Specifying methods and training schemas
    training_schema = "mts"
    method = "NewTimesNet"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "window_size": 48,     # 增大窗口以捕获更多时序信息
            "batch_size": 128,     # 增大batch_size提高训练效率
            "epochs": 20,          # 增加训练轮数
            "learning_rate": 0.001  # 使用较大的学习率配合OneCycleLR
        },
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
 