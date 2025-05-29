from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# 添加MTS-Mixers路径到sys.path
mtsmixer_path = os.path.abspath("EasyTSAD/Methods/MTS-Mixers-main")
if mtsmixer_path not in sys.path:
    sys.path.append(mtsmixer_path)

from EasyTSAD.Controller import TSADController

# 导入MTS-Mixers的必要模块
try:
    from models.MTSMixer import Model as MTSMixerModel
    from layers.Invertible import RevIN
    from layers.Projection import ChannelProjection
    print("[LOG] MTS-Mixers模块导入成功")
except ImportError as e:
    print(f"[ERROR] 导入MTS-Mixers模块失败: {e}")
    print("[LOG] 使用fallback实现")
    # Fallback: 如果导入失败，使用本地实现
    MTSMixerModel = None

# ===================== MTS-Mixers 模型组件 (内嵌实现) =====================

class RevIN(nn.Module):
    """可逆实例标准化"""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class ChannelProjection(nn.Module):
    """通道投影层"""
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))
            x = torch.stack(x_out, dim=-1)
        else: 
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)
        return x


class MLPBlock(nn.Module):
    """MLP块"""
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    """因子化时间混合"""
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()
        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad
        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))
        x = self.merge(x.shape, x_samp)
        return x


class FactorizedChannelMixing(nn.Module):
    """因子化通道混合"""
    def __init__(self, input_dim, factorized_dim):
        super().__init__()
        # 确保factorized_dim不超过input_dim
        factorized_dim = min(factorized_dim, input_dim - 1)
        if factorized_dim <= 0:
            factorized_dim = max(1, input_dim // 2)
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    """Mixer块"""
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, 
                 fac_T, fac_C, sampling, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        
        # 修复通道混合的参数
        if fac_C and channels_dim > 1:
            # 确保factorized维度合理
            factorized_dim = min(channels_hidden_dim, max(1, channels_dim // 2))
            self.channels_mixing = FactorizedChannelMixing(channels_dim, factorized_dim)
        else:
            self.channels_mixing = None
            
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class MTSMixerModel(nn.Module):
    """MTS-Mixers 主模型"""
    def __init__(self, seq_len, enc_in, pred_len, d_model=512, d_ff=2048, e_layers=2, 
                 norm=True, rev=False, fac_T=False, fac_C=False, sampling=2, individual=False):
        super().__init__()
        
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(seq_len, enc_in, d_model, d_ff, fac_T, fac_C, sampling, norm) 
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(enc_in) if norm else None
        self.projection = ChannelProjection(seq_len, pred_len, enc_in, individual)
        self.rev = RevIN(enc_in) if rev else None

    def forward(self, x):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        x = self.rev(x, 'denorm') if self.rev else x

        return x


# ===================== 主程序 =====================

if __name__ == "__main__":
    
    print("[LOG] 开始运行runMTSMixer.py")
    
    # Create a global controller
    gctrl = TSADController()
    print("[LOG] TSADController已创建")
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["machine-1", "machine-2", "machine-3"]
    # datasets = ["TODS"]
    dataset_types = "MTS"
    #
    # set datasets path, dirname is the absolute/relative path of dataset.
    
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

    print("[LOG] 开始定义MTSMixer类")
    
    class MTSMixer(BaseMethod):
        def __init__(self, params: dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            print(f"[LOG] MTSMixer.__init__() 调用，使用设备: {self.device}")
            
        def _build_model(self, input_dim, seq_len=96):
            """构建MTS-Mixers模型"""
            print(f"[LOG] 构建模型，输入维度: {input_dim}, 序列长度: {seq_len}")
            
            # 根据数据规模调整模型参数，确保参数合理
            d_model = min(128, max(32, input_dim * 2))  # 减小模型规模
            d_ff = min(256, max(64, input_dim * 3))     # 减小d_ff，确保不会太大
            
            self.model = MTSMixerModel(
                seq_len=seq_len,
                enc_in=input_dim,
                pred_len=seq_len,  # 重构相同长度用于异常检测
                d_model=d_model,
                d_ff=d_ff,
                e_layers=2,
                norm=True,
                rev=True,  # 使用RevIN进行标准化
                fac_T=True,  # 使用因子化时间混合
                fac_C=True if input_dim > 4 else False,  # 当通道数较多时使用因子化通道混合
                sampling=2,
                individual=False
            ).to(self.device)
            
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"[LOG] MTS-Mixers模型构建成功，参数数量: {param_count}")
            
        def train_valid_phase(self, tsData):
            print(f"[LOG] MTSMixer.train_valid_phase() 调用，数据形状: {tsData.train.shape}")
            
            train_data = torch.FloatTensor(tsData.train).to(self.device)
            seq_len, input_dim = train_data.shape
            
            # 设置窗口大小
            window_size = min(seq_len, 96)  # 最大96，确保不超过数据长度
            
            # 构建模型
            self._build_model(input_dim, window_size)
            
            # 创建滑动窗口数据用于训练
            train_windows = []
            step_size = max(1, window_size // 4)  # 窗口步长
            
            for i in range(0, max(1, seq_len - window_size + 1), step_size):
                window = train_data[i:i+window_size, :]
                if window.shape[0] == window_size:
                    train_windows.append(window)
            
            # 如果没有足够的数据创建窗口
            if len(train_windows) == 0:
                if seq_len < window_size:
                    # 对短序列进行填充
                    padded_data = torch.zeros(window_size, input_dim).to(self.device)
                    padded_data[:seq_len, :] = train_data
                    train_windows = [padded_data]
                else:
                    train_windows = [train_data[:window_size]]
                    
            train_tensor = torch.stack(train_windows)  # [batch_size, seq_len, input_dim]
            print(f"[LOG] 训练数据形状: {train_tensor.shape}")
            
            # 训练模型
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            num_epochs = 100
            batch_size = min(16, train_tensor.shape[0])
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            print(f"[LOG] 开始训练，epochs: {num_epochs}, batch_size: {batch_size}")
            
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0
                
                # 随机打乱数据
                indices = torch.randperm(train_tensor.shape[0])
                
                for i in range(0, train_tensor.shape[0], batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch = train_tensor[batch_indices]
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    output = self.model(batch)
                    loss = criterion(output, batch)
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / max(1, num_batches)
                
                # 早停机制
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"[LOG] 早停在第 {epoch+1} 轮，最佳损失: {best_loss:.6f}")
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f"[LOG] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            print("[LOG] 模型训练完成")
            
        def test_phase(self, tsData: MTSData):
            print(f"[LOG] MTSMixer.test_phase() 调用，测试数据形状: {tsData.test.shape}")
            
            if self.model is None:
                print("[ERROR] 模型未训练，无法进行测试")
                return
                
            test_data = torch.FloatTensor(tsData.test).to(self.device)
            seq_len, input_dim = test_data.shape
            window_size = 96  # 使用固定窗口大小
            
            self.model.eval()
            scores = []
            
            print(f"[LOG] 开始异常检测，序列长度: {seq_len}, 窗口大小: {window_size}")
            
            with torch.no_grad():
                for i in range(seq_len):
                    # 获取当前时间点的窗口
                    start_idx = max(0, i - window_size + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx < window_size:
                        # 对于序列开始部分，用零填充或重复填充
                        window = torch.zeros(window_size, input_dim).to(self.device)
                        actual_data = test_data[start_idx:end_idx, :]
                        window[-actual_data.shape[0]:, :] = actual_data
                        target_idx = window_size - 1
                    else:
                        window = test_data[start_idx:end_idx, :]
                        target_idx = window_size - 1
                    
                    window_batch = window.unsqueeze(0)  # [1, seq_len, input_dim]
                    
                    # 模型预测
                    try:
                        reconstructed = self.model(window_batch)
                        # 计算重构误差
                        reconstruction_error = torch.mean((window_batch - reconstructed) ** 2, dim=2)  # [1, seq_len]
                        score = reconstruction_error[0, target_idx].item()
                    except Exception as e:
                        print(f"[WARNING] 模型预测出错: {e}")
                        score = 0.0
                    
                    scores.append(score)
            
            scores = np.array(scores)
            
            # 平滑处理
            if len(scores) > 5:
                window_size_smooth = min(5, len(scores))
                smoothed_scores = np.convolve(scores, np.ones(window_size_smooth)/window_size_smooth, mode='same')
                scores = smoothed_scores
            
            # 标准化分数到[0,1]
            if len(scores) > 0:
                score_min, score_max = np.min(scores), np.max(scores)
                if score_max > score_min:
                    scores = (scores - score_min) / (score_max - score_min)
                else:
                    scores = np.zeros_like(scores)
            
            self.__anomaly_score = scores
            print(f"[LOG] 异常分数计算完成，长度: {len(scores)}, 范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            print(f"[LOG] MTSMixer.param_statistic() 调用，保存到: {save_file}")
            model_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
            param_info = f"""MTSMixer异常检测模型信息:
                模型类型: MTS-Mixers (用于多元时序异常检测)
                使用设备: {self.device}
                模型参数数量: {model_params}
                模型架构: MTS-Mixers with RevIN normalization
                特征: 
                - 因子化时间混合 (Factorized Temporal Mixing)
                - 可选因子化通道混合 (Factorized Channel Mixing)  
                - 可逆实例标准化 (RevIN)
                - 基于重构误差的异常检测
                """
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(param_info)
    
    print("[LOG] MTSMixer类定义完成")
    
    """============= Run your algo. ============="""
    training_schema = "mts"
    method = "MTSMixer"
    
    print(f"[LOG] 开始运行实验，method={method}, training_schema={training_schema}")
    
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        preprocess="z-score",  # 使用z-score标准化
    )
    print("[LOG] 实验运行完成")
       
    """============= [EVALUATION SETTINGS] ============="""
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
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
    print("[LOG] 开始绘图")
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
    print("[LOG] 绘图完成")
    
    print("[LOG] runMTSMixer.py执行完毕")
