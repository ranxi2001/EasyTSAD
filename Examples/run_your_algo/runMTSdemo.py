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
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData

# =============== TimesNet 组件实现 ===============

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
    def __init__(self, seq_len=96, pred_len=0, enc_in=1, c_out=1, d_model=64, d_ff=64,
                 e_layers=2, top_k=5, num_kernels=6, embed='timeF', freq='h', dropout=0.1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.embed = embed
        self.freq = freq
        self.dropout = dropout

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

    """============= 实现 TimesNet 算法 ============="""

    class MTSTimesNet(BaseMethod):  
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # TimesNet 参数配置
            self.window_size = params.get('window_size', 96)
            self.batch_size = params.get('batch_size', 32)
            self.epochs = params.get('epochs', 10)
            self.learning_rate = params.get('learning_rate', 0.001)
            
            print(f"使用设备: {self.device}")
            
        def train_valid_phase(self, tsData):
            print(f"训练数据形状: {tsData.train.shape}")
            
            # 动态确定输入维度
            enc_in = tsData.train.shape[1] if len(tsData.train.shape) > 1 else 1
            
            # 创建配置
            self.config = TimesNetConfig(
                seq_len=self.window_size,
                enc_in=enc_in,
                c_out=enc_in
            )
            
            # 创建模型
            self.model = TimesNetModel(self.config).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
            
            # 准备训练数据
            if len(tsData.train.shape) == 1:
                train_data = tsData.train.reshape(-1, 1)
            else:
                train_data = tsData.train
                
            train_dataset = MTSTimeSeriesDataset(train_data, self.window_size)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # 训练模型
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = 0
                
                progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
                for batch_x in progress_bar:
                    batch_x = batch_x.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_x)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
            
        def test_phase(self, tsData: MTSData):
            print(f"测试数据形状: {tsData.test.shape}")
            
            # 准备测试数据
            if len(tsData.test.shape) == 1:
                test_data = tsData.test.reshape(-1, 1)
            else:
                test_data = tsData.test
                
            test_dataset = MTSTimeSeriesDataset(test_data, self.window_size)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # 测试模型
            self.model.eval()
            all_scores = []
            
            with torch.no_grad():
                progress_bar = tqdm.tqdm(test_loader, desc='Testing')
                for batch_x in progress_bar:
                    batch_x = batch_x.to(self.device)
                    outputs = self.model(batch_x)
                    
                    # 计算重构误差作为异常分数
                    mse = torch.mean((batch_x - outputs) ** 2, dim=(1, 2))
                    scores = mse.cpu().numpy()
                    all_scores.extend(scores)
            
            # 处理分数长度，确保与原始数据长度一致
            scores = np.array(all_scores)
            
            # 为前面的时间步填充平均分数
            if len(scores) < len(test_data):
                pad_length = len(test_data) - len(scores)
                avg_score = np.mean(scores) if len(scores) > 0 else 0
                padded_scores = np.concatenate([np.full(pad_length, avg_score), scores])
                scores = padded_scores
            
            # 归一化分数
            if len(scores) > 0 and np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            
            self.__anomaly_score = scores
            print(f"异常分数计算完成，长度: {len(scores)}")
            
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
    method = "MTSTimesNet"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "window_size": 64,
            "batch_size": 16,
            "epochs": 5,
            "learning_rate": 0.001,
        },
        # use which method to preprocess original data. 
        # Default: raw
        # Option: 
        #   - z-score(Standardlization), 
        #   - min-max(Normalization), 
        #   - raw (original curves)
        preprocess="z-score",
    )
       
        
    """============= [EVALUATION SETTINGS] ============="""

    print("[LOG] 开始设置评估协议")    
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
    print("[LOG] 开始绘图")
    # plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
