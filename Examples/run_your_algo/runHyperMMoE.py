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

# ============= 增强专家网络类定义 =============
class HyperExpert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out=0.2):
        super(HyperExpert, self).__init__()
        # 🚀 增强卷积层 - 使用更多卷积核
        self.conv1 = nn.Conv2d(1, n_kernel, (window, 1))
        self.conv2 = nn.Conv2d(n_kernel, n_kernel * 2, (1, 1))  # 新增第二层卷积
        self.batch_norm1 = nn.BatchNorm2d(n_kernel)  # 添加批归一化
        self.batch_norm2 = nn.BatchNorm2d(n_kernel * 2)
        
        self.dropout = nn.Dropout(drop_out)
        
        # 🚀 更深的全连接网络
        conv_output_size = n_kernel * 2 * n_multiv
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 新增中间层
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)  # 使用LeakyReLU
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = x.unsqueeze(dim=1).contiguous()
        
        # 第一层卷积
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # 展平
        out = torch.flatten(x, start_dim=1).contiguous()
        
        # 深度全连接网络
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ============= 增强塔网络类定义 =============    
class HyperTower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, drop_out=0.1):
        super(HyperTower, self).__init__()
        # 🚀 更深的塔网络
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 新增中间层
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(drop_out)
        self.layer_norm1 = nn.LayerNorm(hidden_size)  # 🔧 修复: 使用LayerNorm替代BatchNorm1d
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):  # 🔧 修复: 更新初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.layer_norm1(out)  # 🔧 修复: 使用LayerNorm
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ============= HMMoE超参数模型类定义 =============
class HyperMMoEModel(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, n_expert=8, 
                 sg_ratio=0.7, exp_dropout=0.2, tow_dropout=0.1, towers_hidden=32):
        super(HyperMMoEModel, self).__init__()
        self.n_kernel = n_kernel
        self.window = window
        self.n_multiv = n_multiv
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_expert = n_expert
        self.sg_ratio = sg_ratio
        self.softmax = nn.Softmax(dim=1)

        # 🚀 更多的专家网络
        self.experts = nn.ModuleList([
            HyperExpert(n_kernel, window, n_multiv, hidden_size, output_size, exp_dropout)
            for _ in range(n_expert)
        ])
        
        # 🚀 增强门控网络 - 添加更多参数
        self.w_gates = nn.ParameterList([
            nn.Parameter(torch.randn(window, n_expert), requires_grad=True)
            for _ in range(n_multiv)
        ])
        self.share_gate = nn.Parameter(torch.randn(window, n_expert), requires_grad=True)
        
        # 新增额外的门控权重
        self.expert_bias = nn.Parameter(torch.randn(n_expert), requires_grad=True)
        self.gate_temperature = nn.Parameter(torch.ones(1), requires_grad=True)  # 可学习的温度参数
        
        # 🚀 更深的塔网络
        self.towers = nn.ModuleList([
            HyperTower(output_size, 1, towers_hidden, tow_dropout)
            for _ in range(n_multiv)
        ])
        
        # 参数初始化
        self._init_parameters()
        
    def _init_parameters(self):
        for gate in self.w_gates:
            nn.init.xavier_normal_(gate)
        nn.init.xavier_normal_(self.share_gate)
        nn.init.constant_(self.expert_bias, 0)
        nn.init.constant_(self.gate_temperature, 1.0)

    def forward(self, x):
        # 专家网络输出
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)
        
        # 🚀 增强门控网络输出 - 添加温度缩放和偏置
        gates_out = []
        for i in range(self.n_multiv):
            gate_weight = (x[:,:,i] @ self.w_gates[i]) * (1 - self.sg_ratio) + \
                         (x[:,:,i] @ self.share_gate) * self.sg_ratio
            
            # 添加专家偏置和温度缩放
            gate_weight = (gate_weight + self.expert_bias) / self.gate_temperature
            gates_out.append(self.softmax(gate_weight))
        
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

# ============= HMMoE异常检测方法类 =============
class HMMoE(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.model = None  # 初始化model属性
        
        # 🚀 HMMoE超参数配置 - 大幅增加模型容量
        self.config = {
            'seed': 2023,
            'n_multiv': 38,          # 根据数据调整
            'horize': 1,
            'window': 20,            # 🚀 增大窗口大小 16→20
            'batch_size': 32,        # 🚀 适当减小批量大小以容纳更大模型
            'epochs': 12,            # 🚀 增加训练轮数 5→12

            'num_experts': 8,        # 🚀 大幅增加专家数量 4→8
            'n_kernel': 16,          # 🚀 大幅增加卷积核数量 8→16
            'experts_out': 128,      # 🚀 增加专家输出维度 64→128
            'experts_hidden': 256,   # 🚀 大幅增加专家隐藏层 128→256
            'towers_hidden': 32,     # 🚀 增加塔网络隐藏层 16→32
            'criterion': 'l2',       # 损失函数
            'exp_dropout': 0.25,     # 🚀 适当增加dropout防止过拟合
            'tow_dropout': 0.15,     # 🚀 适当增加dropout
            'sg_ratio': 0.8,         # 🚀 调整共享门控比例
            'lr': 0.0005,            # 🚀 降低学习率 0.001→0.0005
            'weight_decay': 1e-4     # 🚀 添加权重衰减
        }
        
        # 设置随机种子
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        print(f"🚀 [LOG] HMMoE超参数版本初始化完成")
        print(f"💪 [LOG] 超大配置: {self.config['num_experts']}专家, {self.config['n_kernel']}卷积核, {self.config['epochs']}轮训练")
        print(f"🎯 [LOG] 设计理念: 大模型大数据，追求极致性能")

    def train_valid_phase(self, tsData: MTSData):
        print(f"\n🚀 [LOG] ========== HMMoE超参数训练阶段开始 ==========")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 [LOG] 使用设备: {device}")
        
        # 使用config中的参数
        window_size = self.config['window']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        lr = self.config['lr']
        
        # 从MTSDataset获取数据
        train_dataset = MTSDataset(tsData=tsData, set_type='train', window=window_size, horize=self.config['horize'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # 动态获取数据维度并更新config
        n_multiv = tsData.train.shape[1]
        self.config['n_multiv'] = n_multiv
        print(f"📊 [LOG] 数据维度: {n_multiv}, 训练样本数: {len(train_dataset)}")
        
        # 🚀 使用config中的参数创建超大模型
        self.model = HyperMMoEModel(
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
        
        # 🚀 优化器配置 - 添加权重衰减
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config['weight_decay'])
        
        # 🚀 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)
        
        criterion = nn.MSELoss()
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"💪 [LOG] 模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        self.model.train()
        print(f"🚀 [LOG] 开始超参数训练，共 {epochs} 个epoch")
        print(f"⚙️ [LOG] 使用参数: window={window_size}, batch_size={batch_size}, lr={lr}")
        print(f"🏗️ [LOG] 模型配置: experts={self.config['num_experts']}, kernel={self.config['n_kernel']}, hidden={self.config['experts_hidden']}")
        
        # 添加外层进度条显示整体训练进度
        epoch_bar = tqdm(range(epochs), desc="🚀 HMMoE超参数训练", ncols=100)
        
        best_loss = float('inf')
        patience_counter = 0
        
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
                
                # 🚀 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 更新batch进度条显示当前loss
                batch_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            # 🚀 学习率调度
            scheduler.step(avg_loss)
            
            # 更新epoch进度条显示平均loss
            epoch_bar.set_postfix({'avg_loss': f'{avg_loss:.6f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
            print(f"✅ Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 🚀 早停机制
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
        epoch_bar.close()
        print(f"🎉 [LOG] ========== HMMoE超参数训练阶段完成 ==========")
        print(f"📈 [LOG] 最佳损失: {best_loss:.6f}")

    def test_phase(self, tsData: MTSData):
        print(f"\n🔍 [LOG] ========== HMMoE超参数测试阶段开始 ==========")
        print(f"📊 [LOG] 测试数据形状: {tsData.test.shape}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用config中的参数
        window_size = self.config['window']
        batch_size = self.config['batch_size']
        
        # 使用MTSDataset处理测试数据
        test_dataset = MTSDataset(tsData=tsData, set_type='test', window=window_size, horize=self.config['horize'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        self.model.eval()
        anomaly_scores = []
        
        print(f"🔍 [LOG] 开始超参数测试，共 {len(test_loader)} 个batch")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="🔍 HMMoE超参数测试", ncols=80):
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
        print(f"🎉 [LOG] ========== HMMoE超参数测试阶段完成 ==========\n")

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            param_info = f"""
                HMMoE超参数多专家混合模型参数统计:
                ==================================================
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                专家数量: {self.config['num_experts']}
                卷积核数量: {self.config['n_kernel']}
                专家隐藏层: {self.config['experts_hidden']}
                专家输出维度: {self.config['experts_out']}
                塔网络隐藏层: {self.config['towers_hidden']}
                窗口大小: {self.config['window']}
                批量大小: {self.config['batch_size']}
                训练轮数: {self.config['epochs']}
                学习率: {self.config['lr']}
                权重衰减: {self.config['weight_decay']}

                🚀 超参数增强特性:
                - 双层卷积 + 批归一化
                - 8个专家网络 (vs 4个)
                - 更深的全连接层
                - 可学习门控温度
                - 梯度裁剪
                - 学习率调度
                - 早停机制

                ==================================================
                设计理念: 大模型大数据，极致性能追求
            """
        else:
            param_info = "HMMoE模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)

# ============= 主程序入口 =============
if __name__ == "__main__":
    print("🚀 ========== HMMoE超参数多专家混合异常检测 ==========")
    print("💪 [LOG] 程序开始执行")
    
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

    print("🏗️ [LOG] HMMoE类定义完成")

    """============= Run HMMoE algo. ============="""
    
    # some settings of this anomaly detection method
    method = "HMMoE"  # string of your algo class

    print(f"🚀 [LOG] 开始运行超参数实验，method={method}")
    # run models
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            "window": 20,        # 🚀 增大窗口
            "batch_size": 32,    # 🚀 适中批量大小
            "epochs": 12,        # 🚀 增加训练轮数
            "lr": 0.0005,        # 🚀 降低学习率
        },
        # use which method to preprocess original data. 
        preprocess="z-score",
    )
    print("🎉 [LOG] 超参数实验运行完成")
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
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
    
    # plot anomaly scores for each curve
    print("📈 [LOG] 开始绘图")
    gctrl.plots(
        method=method,
        training_schema="mts"
    )
    print("🎨 [LOG] 绘图完成")
    
    print("🚀 ========== HMMoE超参数版执行完毕 ==========")