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

# ============= 高效数据集类定义 =============
class FastMTSDataset(torch.utils.data.Dataset):
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
        
        # 数据采样策略 - 如果数据太大就采样
        if self.sample_num > 30000:  # 限制训练样本数量
            indices = np.random.choice(self.sample_num, 30000, replace=False)
            indices.sort()
            self.selected_indices = indices
            self.sample_num = len(indices)
            print(f"🔧 [LOG] 数据采样: {len(indices)} 样本")
        else:
            self.selected_indices = None
        
        # 预计算所有样本
        self.samples, self.labels = self.__precompute_samples(rawdata)

    def __precompute_samples(self, data):
        if self.selected_indices is not None:
            actual_sample_num = len(self.selected_indices)
        else:
            actual_sample_num = self.sample_num
            
        X = torch.zeros((actual_sample_num, self.window, self.var_num), dtype=torch.float32)
        Y = torch.zeros((actual_sample_num, 1, self.var_num), dtype=torch.float32)

        for idx in range(actual_sample_num):
            if self.selected_indices is not None:
                i = self.selected_indices[idx]
            else:
                i = idx
                
            start = i
            end = i + self.window
            X[idx, :, :] = torch.from_numpy(data[start:end, :]).float()
            Y[idx, :, :] = torch.from_numpy(data[end+self.horize-1, :]).float()

        return X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return self.samples[idx, :, :], self.labels[idx, :, :]

# ============= 简化快速专家网络 =============
class FastExpert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out=0.1):
        super(FastExpert, self).__init__()
        # 使用简单的线性层，不用卷积
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x shape: [batch, window, features] -> flatten
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)

# ============= 简化门控网络 =============    
class FastGate(nn.Module):
    def __init__(self, input_size, num_experts):
        super(FastGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 权重初始化
        nn.init.xavier_normal_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 0)
        
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.gate(x_flat)

# ============= 简化塔网络 =============
class FastTower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(FastTower, self).__init__()
        self.tower = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.tower(x)

# ============= NewLMMoE快速优化模型 =============
class FastNewLMMoEModel(nn.Module):
    def __init__(self, window_size, n_features, num_experts=3, expert_hidden=64, expert_output=32, tower_hidden=16):
        super(FastNewLMMoEModel, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.num_experts = num_experts
        self.expert_output = expert_output
        
        input_size = window_size * n_features
        
        # 快速专家网络
        self.experts = nn.ModuleList([
            FastExpert(input_size, expert_hidden, expert_output)
            for _ in range(num_experts)
        ])
        
        # 简化门控网络 - 为每个特征维度创建门控
        self.gates = nn.ModuleList([
            FastGate(input_size, num_experts)
            for _ in range(n_features)
        ])
        
        # 快速塔网络
        self.towers = nn.ModuleList([
            FastTower(expert_output, 1, tower_hidden)
            for _ in range(n_features)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算所有专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch, expert_output]
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch, expert_output, num_experts]
        
        # 为每个特征维度计算输出
        tower_outputs = []
        for i in range(self.n_features):
            # 计算门控权重
            gate_weights = self.gates[i](x)  # [batch, num_experts]
            gate_weights = gate_weights.unsqueeze(1)  # [batch, 1, num_experts]
            
            # 加权组合专家输出
            combined_output = torch.sum(expert_outputs * gate_weights, dim=2)  # [batch, expert_output]
            
            # 通过塔网络
            tower_output = self.towers[i](combined_output)  # [batch, 1]
            tower_outputs.append(tower_output)
        
        # 合并输出
        final_output = torch.stack(tower_outputs, dim=2)  # [batch, 1, n_features]
        
        return final_output

# ============= NewLMMoE快速异常检测方法类 =============
class NewLMMoE(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.model = None
        
        # NewLMMoE快速配置 - 参考CAD的高效设置
        self.config = {
            'window': 16,
            'batch_size': 64,        # 保持较大批量
            'epochs': 3,             # 🚀 进一步减少训练轮数
            'lr': 0.001,
            
            'num_experts': 3,        # 保持3个专家
            'expert_hidden': 48,     # 🔧 进一步减少 96→48
            'expert_output': 24,     # 🔧 减少输出维度 32→24
            'tower_hidden': 16,      # 保持塔网络大小
        }
        
        print(f"🚀 [LOG] NewLMMoE快速版本初始化完成")
        print(f"⚡ [LOG] 高效配置: {self.config['num_experts']}专家, {self.config['epochs']}轮训练")
        print(f"🎯 [LOG] 设计理念: 简化架构，极速训练，保持性能")

    def train_valid_phase(self, tsData: MTSData):
        print(f"\n⚡ [LOG] ========== NewLMMoE快速训练开始 ==========")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 [LOG] 使用设备: {device}")
        
        # 快速数据加载
        train_dataset = FastMTSDataset(tsData=tsData, set_type='train', 
                                      window=self.config['window'], horize=1)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0,  # 避免多线程问题
            pin_memory=False  # 简化内存管理
        )
        
        n_features = tsData.train.shape[1]
        print(f"📊 [LOG] 数据维度: {n_features}, 训练样本数: {len(train_dataset)}")
        
        # 创建快速模型
        self.model = FastNewLMMoEModel(
            window_size=self.config['window'],
            n_features=n_features,
            num_experts=self.config['num_experts'],
            expert_hidden=self.config['expert_hidden'],
            expert_output=self.config['expert_output'],
            tower_hidden=self.config['tower_hidden']
        ).to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        criterion = nn.MSELoss()
        
        self.model.train()
        print(f"⚡ [LOG] 开始快速训练，共 {self.config['epochs']} 个epoch")
        print(f"🏗️ [LOG] 模型参数: experts={self.config['num_experts']}, expert_hidden={self.config['expert_hidden']}")
        
        # 快速训练循环
        for epoch in range(self.config['epochs']):
            total_loss = 0
            batch_count = 0
            
            progress_bar = tqdm(train_loader, desc=f"⚡ Epoch {epoch+1}/{self.config['epochs']}", ncols=80)
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"✅ Epoch {epoch+1}/{self.config['epochs']}, Average Loss: {avg_loss:.6f}")
        
        print(f"🎉 [LOG] ========== NewLMMoE快速训练完成 ==========\n")

    def test_phase(self, tsData: MTSData):
        print(f"\n🔍 [LOG] ========== NewLMMoE快速测试开始 ==========")
        print(f"📊 [LOG] 测试数据形状: {tsData.test.shape}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 快速测试数据加载
        test_dataset = FastMTSDataset(tsData=tsData, set_type='test', 
                                     window=self.config['window'], horize=1)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        anomaly_scores = []
        
        print(f"🔍 [LOG] 开始快速测试，共 {len(test_loader)} 个batch")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="🔍 快速测试", ncols=80):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                mse = torch.mean((output - target) ** 2, dim=(1, 2))
                anomaly_scores.extend(mse.cpu().numpy())
        
        # 处理异常分数长度
        scores = np.array(anomaly_scores)
        full_scores = np.zeros(len(tsData.test))
        
        # 填充前面的时间步
        if len(scores) < len(tsData.test):
            pad_length = len(tsData.test) - len(scores)
            avg_score = np.mean(scores) if len(scores) > 0 else 0
            full_scores = np.concatenate([np.full(pad_length, avg_score), scores])
        else:
            full_scores = scores[:len(tsData.test)]
        
        # 归一化
        if len(full_scores) > 0 and np.max(full_scores) > np.min(full_scores):
            full_scores = (full_scores - np.min(full_scores)) / (np.max(full_scores) - np.min(full_scores))
        
        self.__anomaly_score = full_scores
        print(f"🎉 [LOG] ========== NewLMMoE快速测试完成 ==========\n")

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            param_info = f"""
                NewLMMoE快速版本模型参数统计:
                ==================================================
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                专家数量: {self.config['num_experts']}
                专家隐藏层: {self.config['expert_hidden']}
                专家输出维度: {self.config['expert_output']}
                塔网络隐藏层: {self.config['tower_hidden']}
                窗口大小: {self.config['window']}
                批量大小: {self.config['batch_size']}
                训练轮数: {self.config['epochs']}
                学习率: {self.config['lr']}

                ⚡ 快速优化特性:
                - 线性层架构 (去掉卷积)
                - 数据采样加速
                - 简化门控机制
                - 轻量化专家网络
                - 快速训练策略

                ==================================================
                设计理念: 简化架构 + 极速训练 + 保持性能
            """
        else:
            param_info = "NewLMMoE模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)

# ============= 主程序入口 =============
if __name__ == "__main__":
    print("⚡ ========== NewLMMoE轻量级多专家混合异常检测快速版 ==========")
    print("🚀 [LOG] 程序开始执行")
    
    # Create a global controller
    gctrl = TSADController()
    print("🔧 [LOG] TSADController已创建")
    
    # Set dataset
    datasets = ["machine-1", "machine-2", "machine-3"]
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("📊 [LOG] 数据集设置完成")

    print("🏗️ [LOG] NewLMMoE类定义完成")

    """============= Run NewLMMoE Fast algo. ============="""
    
    method = "NewLMMoE"

    print(f"⚡ [LOG] 开始运行快速实验，method={method}")
    # run models with fast hyperparameters
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            "window": 16,
            "batch_size": 64,
            "epochs": 3,         # ⚡ 超快训练
            "lr": 0.001,
        },
        preprocess="z-score",
    )
    print("🎉 [LOG] 快速实验运行完成")
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    print("📊 [LOG] 开始设置评估协议")
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )
    print("✅ [LOG] 评估协议设置完成")

    print("🔍 [LOG] 开始执行评估")
    gctrl.do_evals(
        method=method,
        training_schema="mts"
    )
    print("🎉 [LOG] 评估执行完成")
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    print("📈 [LOG] 开始绘图")
    gctrl.plots(
        method=method,
        training_schema="mts"
    )
    print("🎨 [LOG] 绘图完成")
    
    print("⚡ ========== NewLMMoE快速版执行完毕 ==========")