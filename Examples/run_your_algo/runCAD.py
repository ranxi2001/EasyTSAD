from typing import Dict
import numpy as np
import os
from random import shuffle
from multiprocessing import cpu_count
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData
import tqdm

try:
    from torchinfo import summary
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, Callback
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    print("警告: 某些可选依赖未安装，将使用简化版本")
    # 创建简化的替代类
    class TensorBoardLogger:
        def __init__(self, name="logs", save_dir="./"):
            pass
    
    class EarlyStopping:
        def __init__(self, monitor='val_loss', patience=5, verbose=True, mode='min'):
            pass
    
    class ModelCheckpoint:
        def __init__(self, monitor='val_loss', save_top_k=1, mode='min'):
            pass

import warnings
warnings.filterwarnings("ignore")


class MTSDataset(torch.utils.data.Dataset):
    
    def __init__(self, tsData: MTSData, set_type:str, window: int, horize: int):

        assert type(set_type) == type('str')
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

class Expert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out):
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
    
class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_out):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class SimpleMMoE(nn.Module):
    """简化版本的MMoE模型，不依赖pytorch_lightning"""
    def __init__(self, config):
        super(SimpleMMoE, self).__init__()
        self.hp = config 
        self.n_multiv = config['n_multiv']
        self.n_kernel = config['n_kernel']
        self.window = config['window']
        self.num_experts = config['num_experts']
        self.experts_out = config['experts_out']
        self.experts_hidden = config['experts_hidden']
        self.towers_hidden = config['towers_hidden']

        # task num = n_multiv
        self.tasks = config['n_multiv']
        self.criterion = config['criterion']
        self.exp_dropout = config['exp_dropout']
        self.tow_dropout = config['tow_dropout']

        self.softmax = nn.Softmax(dim=1)
        
        self.experts = nn.ModuleList([Expert(self.n_kernel, self.window, self.n_multiv, self.experts_hidden, self.experts_out, self.exp_dropout) \
            for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True) \
            for i in range(self.tasks)])
        self.share_gate = nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True)
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden, self.tow_dropout) \
            for i in range(self.tasks)])
            
    def forward(self, x):
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)
        
        gates_out = [self.softmax((x[:,:,i] @ self.w_gates[i]) * (1 - self.hp['sg_ratio']) + (x[:,:,i] @ self.share_gate) * self.hp['sg_ratio']) for i in range(self.tasks)]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_out_tensor for g in gates_out]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        
        tower_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        tower_output = torch.stack(tower_output, dim=0).permute(1,2,0)
        
        return tower_output

    def loss(self, labels, predictions):
        if self.criterion == "l1":
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == "l2":
            loss = F.mse_loss(predictions, labels)
        return loss


class CAD(BaseMethod):
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")

        self.config = {
            'seed': 2023,
            'n_multiv': 38,  # 根据数据调整
            'horize': 1,
            'window': 16,    # 减小窗口大小提高速度
            'batch_size': 64,
            'epochs': 5,     # 减少训练轮数

            'num_experts': 4,  # 减少专家数量
            'n_kernel': 8,     # 减少卷积核数量
            'experts_out': 64,
            'experts_hidden': 128,
            'towers_hidden': 16,
            'criterion': 'l2',
            'exp_dropout': 0.2,
            'tow_dropout': 0.1,
            'sg_ratio': 0.7,
            'lr': 0.001
        }

        self.seed = self.config['seed']
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.model = None

    def train_valid_phase(self, tsData: MTSData):
        print(f"训练数据形状: {tsData.train.shape}")
        
        # 动态调整n_multiv
        self.config['n_multiv'] = tsData.train.shape[1]
        
        print("构建模型...")
        self.model = SimpleMMoE(self.config).to(self.device)
        
        # 创建数据集
        train_dataset = MTSDataset(tsData=tsData, set_type='train', 
                                 window=self.config['window'], 
                                 horize=self.config['horize'])
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0  # Windows上设为0避免问题
        )
        
        # 训练模型
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(x.float())
                loss = self.model.loss(y.float(), y_pred)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}/{self.config["epochs"]}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")

    def test_phase(self, tsData: MTSData):
        print(f"测试数据形状: {tsData.test.shape}")
        
        # 创建测试数据集
        test_dataset = MTSDataset(tsData=tsData, set_type='test',
                                window=self.config['window'],
                                horize=self.config['horize'])
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        all_scores = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x.float())
                
                # 计算重构误差
                error = torch.mean((y.float() - y_pred) ** 2, dim=(1, 2))
                all_scores.extend(error.cpu().numpy())
        
        scores = np.array(all_scores)
        
        # 处理长度不匹配问题
        if len(scores) < len(tsData.test):
            # 用平均值填充前面的时间步
            fill_length = len(tsData.test) - len(scores)
            avg_score = np.mean(scores) if len(scores) > 0 else 0
            scores = np.concatenate([np.full(fill_length, avg_score), scores])
        
        # 归一化
        if len(scores) > 0 and np.std(scores) > 1e-10:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
        
        self.__anomaly_score = scores
        print(f"异常分数计算完成，长度: {len(scores)}")

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            param_info = f"""
            CAD模型参数统计:
            总参数数量: {total_params:,}
            可训练参数数量: {trainable_params:,}
            """
        else:
            param_info = "CAD模型尚未初始化"
            
        with open(save_file, 'w') as f:
            f.write(param_info)


if __name__ == "__main__":
    
    print("[LOG] 开始运行runCAD.py")
    
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

    """============= Implement CAD algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    print("[LOG] 开始定义CAD相关类")
    
    # CAD数据集类
    class CADDataset(torch.utils.data.Dataset):
        def __init__(self, data: np.ndarray, window_size: int, prediction_horizon: int = 1):
            super().__init__()
            self.data = data
            self.window_size = window_size
            self.horizon = prediction_horizon
            self.sample_num = max(len(data) - window_size - prediction_horizon + 1, 0)
            
        def __len__(self):
            return self.sample_num
        
        def __getitem__(self, index):
            # 输入窗口
            x = torch.from_numpy(self.data[index:index + self.window_size]).float()
            # 预测目标（下一个时间步）
            y = torch.from_numpy(self.data[index + self.window_size:index + self.window_size + self.horizon]).float()
            return x, y

    # 简化的专家网络
    class SimpleExpert(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
            super(SimpleExpert, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            )
            
        def forward(self, x):
            # x shape: [batch, window, features]
            x_flat = x.view(x.size(0), -1)  # flatten
            return self.net(x_flat)

    # 门控网络
    class GateNetwork(nn.Module):
        def __init__(self, input_size, num_experts):
            super(GateNetwork, self).__init__()
            self.gate = nn.Sequential(
                nn.Linear(input_size, num_experts),
                nn.Softmax(dim=1)
            )
            
        def forward(self, x):
            x_flat = x.view(x.size(0), -1)
            return self.gate(x_flat)

    # 简化的CAD模型（多专家混合）
    class SimpleCADModel(nn.Module):
        def __init__(self, window_size, n_features, num_experts=3, hidden_size=64, expert_output_size=32):
            super(SimpleCADModel, self).__init__()
            self.window_size = window_size
            self.n_features = n_features
            self.num_experts = num_experts
            
            input_size = window_size * n_features
            
            # 创建多个专家
            self.experts = nn.ModuleList([
                SimpleExpert(input_size, hidden_size, expert_output_size)
                for _ in range(num_experts)
            ])
            
            # 门控网络
            self.gate = GateNetwork(input_size, num_experts)
            
            # 输出层
            self.output_layer = nn.Linear(expert_output_size, n_features)
            
        def forward(self, x):
            batch_size = x.size(0)
            
            # 计算每个专家的输出
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(x)  # [batch, expert_output_size]
                expert_outputs.append(expert_out)
            
            expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch, expert_output_size, num_experts]
            
            # 计算门控权重
            gate_weights = self.gate(x)  # [batch, num_experts]
            gate_weights = gate_weights.unsqueeze(1)  # [batch, 1, num_experts]
            
            # 加权组合专家输出
            combined_output = torch.sum(expert_outputs * gate_weights, dim=2)  # [batch, expert_output_size]
            
            # 最终输出
            output = self.output_layer(combined_output)  # [batch, n_features]
            
            return output.unsqueeze(1)  # [batch, 1, n_features] 匹配目标形状

    print("[LOG] 开始定义CAD类")
    class CAD(BaseMethod):
        def __init__(self, params: dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # CAD参数配置
            self.window_size = params.get('window_size', 16)
            self.batch_size = params.get('batch_size', 32)
            self.epochs = params.get('epochs', 10)
            self.learning_rate = params.get('learning_rate', 0.001)
            self.num_experts = params.get('num_experts', 3)
            self.hidden_size = params.get('hidden_size', 64)
            
            print(f"[LOG] 使用设备: {self.device}")
            
        def train_valid_phase(self, tsData):
            print(f"[LOG] CAD.train_valid_phase() 调用，数据形状: {tsData.train.shape}")
            
            # 动态确定输入维度
            n_features = tsData.train.shape[1] if len(tsData.train.shape) > 1 else 1
            
            # 创建模型
            self.model = SimpleCADModel(
                window_size=self.window_size,
                n_features=n_features,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
            
            # 准备训练数据
            if len(tsData.train.shape) == 1:
                train_data = tsData.train.reshape(-1, 1)
            else:
                train_data = tsData.train
            
            # 数据采样 - 如果数据太大，随机采样
            if len(train_data) > 10000:
                indices = np.random.choice(len(train_data), 10000, replace=False)
                indices.sort()
                train_data = train_data[indices]
                print(f"[LOG] 数据采样后训练数据形状: {train_data.shape}")
                
            train_dataset = CADDataset(train_data, self.window_size)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0  # Windows设为0
            )
            
            # 训练模型
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = 0
                
                progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
                for batch_x, batch_y in progress_bar:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                avg_loss = total_loss / num_batches
                print(f"[LOG] Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
            
        def test_phase(self, tsData: MTSData):
            print(f"[LOG] CAD.test_phase() 调用，测试数据形状: {tsData.test.shape}")
            
            # 准备测试数据
            if len(tsData.test.shape) == 1:
                test_data = tsData.test.reshape(-1, 1)
            else:
                test_data = tsData.test
                
            test_dataset = CADDataset(test_data, self.window_size)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=0
            )
            
            # 测试模型
            self.model.eval()
            all_reconstruction_errors = []
            
            with torch.no_grad():
                progress_bar = tqdm.tqdm(test_loader, desc='Testing')
                for batch_x, batch_y in progress_bar:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    
                    # 计算重构误差
                    mse = torch.mean((batch_y - outputs) ** 2, dim=(1, 2))
                    scores = mse.cpu().numpy()
                    all_reconstruction_errors.extend(scores)
            
            # 处理分数长度，确保与原始数据长度一致
            scores = np.array(all_reconstruction_errors)
            
            # 为前面的时间步填充分数
            if len(scores) < len(test_data):
                pad_length = len(test_data) - len(scores)
                avg_score = np.mean(scores) if len(scores) > 0 else 0
                padded_scores = np.concatenate([np.full(pad_length, avg_score), scores])
                scores = padded_scores
            
            # 归一化分数
            if len(scores) > 0 and np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            
            self.__anomaly_score = scores
            print(f"[LOG] 异常分数计算完成，长度: {len(scores)}")
            
        def anomaly_score(self) -> np.ndarray:
            print(f"[LOG] CAD.anomaly_score() 调用")
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            print(f"[LOG] CAD.param_statistic() 调用，保存到: {save_file}")
            if hasattr(self, 'model'):
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                param_info = f"""
                CAD模型参数统计:
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                模型配置:
                - window_size: {self.window_size}
                - num_experts: {self.num_experts}
                - hidden_size: {self.hidden_size}
                """
            else:
                param_info = "CAD模型尚未初始化"
                
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    print("[LOG] CAD类定义完成")
    
    """============= Run CAD algo. ============="""
    # Specifying methods and training schemas
    training_schema = "mts"
    method = "CAD"  # string of your algo class
    
    print(f"[LOG] 开始运行实验，method={method}, training_schema={training_schema}")
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "window_size": 16,
            "batch_size": 32,
            "epochs": 8,
            "learning_rate": 0.001,
            "num_experts": 3,
            "hidden_size": 64,
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
    
    print("[LOG] runCAD.py执行完毕")