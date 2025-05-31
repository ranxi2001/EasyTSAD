import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# ============= 数据集类定义 =============
class MTSDataset(torch.utils.data.Dataset):
    def __init__(self, tsData: MTSData, set_type: str, window: int, horize: int):
        assert set_type in ['train', 'test']
        self.set_type = set_type
        self.window = window
        self.horize = horize        
        
        if set_type == "train":
            rawdata = tsData.train
        elif set_type == "test":
            rawdata = tsData.test
        else:
            raise ValueError('Arg "set_type" in MTSDataset() must be one of "train", "test"')

        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horize + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horize-1, :])

        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]
        return sample

# ============= 专家网络类定义 =============
class Expert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out=0.2):
        super(Expert, self).__init__()
        self.conv = nn.Conv2d(1, n_kernel, (window, 1))
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(n_kernel * n_multiv, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(dim=1).contiguous()
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        
        out = torch.flatten(x, start_dim=1).contiguous()
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# ============= 塔网络类定义 =============    
class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16, drop_out=0.1):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# ============= LightMMoE主模型类定义 =============
class SimpleLightMMoEModel(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, n_expert=4, 
                 sg_ratio=0.7, exp_dropout=0.2, tow_dropout=0.1, towers_hidden=16):
        super(SimpleLightMMoEModel, self).__init__()
        self.n_kernel = n_kernel
        self.window = window
        self.n_multiv = n_multiv
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_expert = n_expert
        self.sg_ratio = sg_ratio
        self.softmax = nn.Softmax(dim=1)

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(n_kernel, window, n_multiv, hidden_size, output_size, exp_dropout)
            for _ in range(n_expert)
        ])
        
        # 门控网络
        self.w_gates = nn.ParameterList([
            nn.Parameter(torch.randn(window, n_expert), requires_grad=True)
            for _ in range(n_multiv)
        ])
        self.share_gate = nn.Parameter(torch.randn(window, n_expert), requires_grad=True)
        
        # 塔网络
        self.towers = nn.ModuleList([
            Tower(output_size, 1, towers_hidden, tow_dropout)
            for _ in range(n_multiv)
        ])

    def forward(self, x):
        # 专家网络输出
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)
        
        # 门控网络输出
        gates_out = [
            self.softmax((x[:,:,i] @ self.w_gates[i]) * (1 - self.sg_ratio) + (x[:,:,i] @ self.share_gate) * self.sg_ratio)
            for i in range(self.n_multiv)
        ]
        
        # 门控加权专家输出
        tower_input = [
            g.t().unsqueeze(2).expand(-1, -1, self.output_size) * experts_out_tensor
            for g in gates_out
        ]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        
        # 塔网络输出
        tower_output = [
            t(ti)
            for t, ti in zip(self.towers, tower_input)
        ]
        tower_output = torch.stack(tower_output, dim=0).permute(1,2,0)
        
        return tower_output

# ============= LightMMoE异常检测方法类 =============
class LightMMoE(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.model = None  # 初始化model属性
        
        # LightMMoE轻量化参数配置
        self.config = {
            'seed': 2023,
            'n_multiv': 38,          # 根据数据调整
            'horize': 1,
            'window': 16,            # 减小窗口大小提高速度
            'batch_size': 64,        # 批量大小
            'epochs': 5,             # 减少训练轮数

            'num_experts': 4,        # 减少专家数量
            'n_kernel': 8,           # 减少卷积核数量
            'experts_out': 64,       # 专家输出维度
            'experts_hidden': 128,   # 专家隐藏层维度
            'towers_hidden': 16,     # 塔网络隐藏层维度
            'criterion': 'l2',       # 损失函数
            'exp_dropout': 0.2,      # 专家网络dropout
            'tow_dropout': 0.1,      # 塔网络dropout
            'sg_ratio': 0.7,         # 共享门控比例
            'lr': 0.001              # 学习率
        }
        
        print(f"[LOG] LightMMoE初始化完成")
        print(f"[LOG] 轻量化配置: {self.config['num_experts']}专家, {self.config['n_kernel']}卷积核, {self.config['epochs']}轮训练")

    def train_valid_phase(self, tsData: MTSData):
        print(f"\n[LOG] ========== LightMMoE训练阶段开始 ==========")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[LOG] 使用设备: {device}")
        
        # 使用config中的参数
        window_size = self.config['window']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        lr = self.config['lr']
        
        # 从MTSDataset获取数据
        train_dataset = MTSDataset(tsData=tsData, set_type='train', window=window_size, horize=self.config['horize'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 动态获取数据维度并更新config
        n_multiv = tsData.train.shape[1]
        self.config['n_multiv'] = n_multiv
        print(f"[LOG] 数据维度: {n_multiv}, 训练样本数: {len(train_dataset)}")
        
        # 使用config中的参数创建模型
        self.model = SimpleLightMMoEModel(
            n_kernel=self.config['n_kernel'],
            window=window_size,
            n_multiv=n_multiv,
            hidden_size=self.config['experts_hidden'],
            output_size=self.config['experts_out'],
            n_expert=self.config['num_experts'],
            sg_ratio=self.config['sg_ratio'],
            exp_dropout=self.config['exp_dropout'],
            tow_dropout=self.config['tow_dropout'],
            towers_hidden=self.config['towers_hidden']
        ).to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        print(f"[LOG] 开始训练，共 {epochs} 个epoch")
        print(f"[LOG] 使用参数: window={window_size}, batch_size={batch_size}, lr={lr}")
        print(f"[LOG] 模型参数: experts={self.config['num_experts']}, kernel={self.config['n_kernel']}")
        
        # 添加外层进度条显示整体训练进度
        epoch_bar = tqdm(range(epochs), desc="🚀 LightMMoE训练进度", ncols=100)
        
        for epoch in epoch_bar:
            total_loss = 0
            batch_count = 0
            
            # 内层进度条显示当前epoch的batch进度
            batch_bar = tqdm(train_loader, desc=f"📊 Epoch {epoch+1}/{epochs}", leave=False, ncols=80)
            
            for batch_idx, (data, target) in enumerate(batch_bar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 更新batch进度条显示当前loss
                batch_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            # 更新epoch进度条显示平均loss
            epoch_bar.set_postfix({'avg_loss': f'{avg_loss:.6f}'})
            print(f"✅ Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        epoch_bar.close()
        print(f"[LOG] ========== LightMMoE训练阶段完成 ==========\n")

    def test_phase(self, tsData: MTSData):
        print(f"\n[LOG] ========== LightMMoE测试阶段开始 ==========")
        print(f"[LOG] 测试数据形状: {tsData.test.shape}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用config中的参数
        window_size = self.config['window']
        batch_size = self.config['batch_size']
        
        # 使用MTSDataset处理测试数据
        test_dataset = MTSDataset(tsData=tsData, set_type='test', window=window_size, horize=self.config['horize'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        anomaly_scores = []
        
        print(f"[LOG] 开始测试，共 {len(test_loader)} 个batch")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="🔍 LightMMoE测试进度", ncols=80):
                data, target = data.to(device), target.to(device)
                
                output = self.model(data)
                mse = torch.mean((output - target) ** 2, dim=(1, 2))
                anomaly_scores.extend(mse.cpu().numpy())
        
        # 调整异常分数长度以匹配原始测试数据
        full_scores = np.zeros(len(tsData.test))
        
        # 前window_size个点使用0分数
        full_scores[:window_size] = 0
        
        # 从第window_size个点开始使用实际计算的分数
        for i, score in enumerate(anomaly_scores):
            if i + window_size < len(full_scores):
                full_scores[i + window_size] = score
        
        self.__anomaly_score = full_scores
        print(f"[LOG] ========== LightMMoE测试阶段完成 ==========\n")

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            param_info = f"""
                LightMMoE轻量级多专家混合模型参数统计:
                ==================================================
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                专家数量: {self.config['num_experts']}
                卷积核数量: {self.config['n_kernel']}
                窗口大小: {self.config['window']}
                批量大小: {self.config['batch_size']}
                训练轮数: {self.config['epochs']}
                学习率: {self.config['lr']}
                ==================================================
                轻量化设计理念: 精简架构，提升效率，保持性能
                            """
        else:
            param_info = "LightMMoE模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)

# ============= 主程序入口 =============
if __name__ == "__main__":
    print("🎯 ========== LightMMoE轻量级多专家混合异常检测 ==========")
    print("[LOG] 程序开始执行")
    
    # Create a global controller
    gctrl = TSADController()
    print("[LOG] TSADController已创建")
    
    # Set dataset
    datasets = ["machine-1", "machine-2", "machine-3"]
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    print("[LOG] LightMMoE类定义完成")

    """============= Run LightMMoE algo. ============="""
    
    # some settings of this anomaly detection method
    method = "LightMMoE"  # string of your algo class

    print(f"[LOG] 开始运行实验，method={method}")
    # run models
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            "window": 16,
            "batch_size": 64,
            "epochs": 5,
            "lr": 0.001,
        },
        # use which method to preprocess original data. 
        preprocess="z-score",
    )
    print("[LOG] 实验运行完成")
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    print("[LOG] 开始设置评估协议")
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )
    print("[LOG] 评估协议设置完成")

    print("[LOG] 开始执行评估")
    gctrl.do_evals(
        method=method,
        training_schema="mts"
    )
    print("[LOG] 评估执行完成")
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    print("[LOG] 开始绘图")
    gctrl.plots(
        method=method,
        training_schema="mts"
    )
    print("[LOG] 绘图完成")
    
    print("🎉 ========== LightMMoE执行完毕 ==========")