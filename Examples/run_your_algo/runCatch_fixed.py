import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict

from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import TSData

# 简化的CATCH模型组件
class SimpleRevIN(nn.Module):
    """简化的可逆归一化层"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.std = torch.std(x, dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std
        elif mode == 'denorm':
            return x * self.std + self.mean
        return x

class SimpleTransformer(nn.Module):
    """简化的Transformer编码器"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.transformer(x)

class SimpleCATCHModel(nn.Module):
    """简化的CATCH模型"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_features = configs.n_features
        
        # 组件
        self.revin = SimpleRevIN(configs.n_features)
        self.input_projection = nn.Linear(configs.n_features, configs.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model))
        
        self.transformer = SimpleTransformer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            num_layers=configs.n_layers,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout
        )
        
        self.output_projection = nn.Linear(configs.d_model, configs.n_features)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # 归一化
        x_norm = self.revin(x, mode='norm')
        
        # 投影到模型维度
        x_proj = self.input_projection(x_norm)
        
        # 位置编码
        x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        encoded = self.transformer(x_proj)
        
        # 输出投影
        output = self.output_projection(encoded)
        
        # 反归一化
        output = self.revin(output, mode='denorm')
        
        return output

class SimpleDataset(Dataset):
    """简化的数据集类"""
    def __init__(self, data, seq_len, stride=1):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        
        if len(data) < seq_len:
            raise ValueError(f"Data length {len(data)} is less than sequence length {seq_len}")
            
        self.indices = list(range(0, len(data) - seq_len + 1, stride))
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        return torch.FloatTensor(self.data[start_idx:end_idx])

class CatchConfig:
    """CATCH配置类"""
    def __init__(self, **kwargs):
        # 默认参数
        self.seq_len = kwargs.get('seq_len', 96)
        self.d_model = kwargs.get('d_model', 64)
        self.n_heads = kwargs.get('n_heads', 4)
        self.n_layers = kwargs.get('n_layers', 2)
        self.d_ff = kwargs.get('d_ff', 256)
        self.dropout = kwargs.get('dropout', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.lr = kwargs.get('lr', 1e-3)
        self.patience = kwargs.get('patience', 5)
        self.n_features = None  # 将在训练时设置

if __name__ == "__main__":
    
    # 创建全局控制器
    gctrl = TSADController()
        
    """============= [数据集设置] ============="""
    datasets = ["machine-1", "machine-2", "machine-3"]
    dataset_types = "MTS"
    
    print("[LOG] 开始设置数据集")
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    """============= CATCH算法实现 ============="""

    class Catch(BaseMethod):
        def __init__(self, hparams) -> None:
            super().__init__()
            self.__anomaly_score = None
            
            # 配置
            self.config = CatchConfig(**hparams)
            self.scaler = StandardScaler()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"[LOG] CATCH算法初始化完成，设备: {self.device}")

        def train_valid_phase(self, tsTrain: TSData):
            print(f"[LOG] CATCH开始训练，训练数据形状: {tsTrain.train.shape}")
            
            # 数据预处理
            train_data = tsTrain.train
            valid_data = tsTrain.valid
            
            # 设置特征数
            self.config.n_features = train_data.shape[1]
            
            # 标准化
            self.scaler.fit(train_data)
            train_data_scaled = self.scaler.transform(train_data)
            valid_data_scaled = self.scaler.transform(valid_data)
            
            # 创建数据集和数据加载器
            train_dataset = SimpleDataset(train_data_scaled, self.config.seq_len)
            valid_dataset = SimpleDataset(valid_data_scaled, self.config.seq_len)
            
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
            self.model = SimpleCATCHModel(self.config).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            
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
                    loss.backward()
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
                
                print(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")
                
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
            print("[LOG] CATCH 不支持 all-in-one 模式")
            return

        def test_phase(self, tsData: TSData):
            print(f"[LOG] CATCH开始测试，测试数据形状: {tsData.test.shape}")
            
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
                    
                    # 计算重构误差
                    mse = torch.mean((batch - output) ** 2, dim=(1, 2))  # 对序列长度和特征维度求均值
                    reconstruction_errors.extend(mse.cpu().numpy())
            
            # 处理异常分数长度问题
            # 由于使用滑动窗口，需要为前面的时间步填充分数
            total_length = len(test_data)
            scores = np.zeros(total_length)
            
            # 前seq_len-1个点使用第一个窗口的分数
            if len(reconstruction_errors) > 0:
                scores[:self.config.seq_len-1] = reconstruction_errors[0]
                scores[self.config.seq_len-1:self.config.seq_len-1+len(reconstruction_errors)] = reconstruction_errors
            
            # 标准化异常分数
            if len(scores) > 0 and np.std(scores) > 0:
                scores = (scores - np.mean(scores)) / np.std(scores)
            
            self.__anomaly_score = scores
            print(f"[LOG] 异常分数计算完成，长度: {len(self.__anomaly_score)}")

        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score

        def param_statistic(self, save_file):
            if hasattr(self, 'model'):
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                param_info = f"简化CATCH模型\n"
                param_info += f"总参数量: {total_params}\n"
                param_info += f"配置:\n"
                for key, value in self.config.__dict__.items():
                    param_info += f"  {key}: {value}\n"
            else:
                param_info = "模型未初始化\n"
                
            with open(save_file, 'w') as f:
                f.write(param_info)

    """============= [算法运行] ============="""
    training_schema = "naive"
    method = "Catch"
    
    # 运行模型
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "seq_len": 100,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 256,
            "dropout": 0.1,
            "batch_size": 64,
            "num_epochs": 10,
            "lr": 1e-3,
            "patience": 5,
        },
        preprocess="z-score",
    )

    """============= [评估设置] ============="""
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    gctrl.set_evals([
        PointF1PA(),
        EventF1PA(),
        EventF1PA(mode="squeeze")
    ])

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )

    """============= [绘图设置] ============="""
    gctrl.plots(
        method=method,
        training_schema=training_schema
    ) 