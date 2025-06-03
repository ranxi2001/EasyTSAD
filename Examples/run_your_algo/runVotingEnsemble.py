import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    print("[LOG] 开始运行VotingEnsemble - 统计学+深度学习投票融合异常检测")
    
    # Create a global controller
    gctrl = TSADController()
    print("[LOG] TSADController已创建")
        
    """============= [DATASET SETTINGS] ============="""
    datasets = ["machine-1", "machine-2", "machine-3"]
    dataset_types = "MTS"
    
    print("[LOG] 开始设置数据集")
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    """============= VotingEnsemble投票融合模型 ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    print("[LOG] 开始定义VotingEnsemble类")
    
    class VotingEnsemble(BaseMethod):
        """
        VotingEnsemble: 统计学+深度学习投票融合异常检测
        
        核心设计理念:
        1. 保持MTSExample统计学方法的强劲Point F1性能 (93.66%)
        2. 添加轻量级深度学习模型专门优化Event连续性
        3. 通过智能投票机制融合两者优势
        4. 专门针对Event F1设计投票策略
        
        设计目标: Point F1维持93%+, Event F1提升到80%+
        投票策略: 统计学主导Point精度，深度学习增强Event连续性
        """
        
        def __init__(self, params: dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            
            # 核心超参数
            self.window_size = 16  # 轻量级窗口
            self.input_dim = 38    # 机器数据特征维度
            self.hidden_dim = 32   # 轻量级隐藏维度
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 初始化模型组件
            self._init_models()
            print("[LOG] VotingEnsemble.__init__() 调用，双路径模型初始化完成")
            
        def _init_models(self):
            """初始化统计学和深度学习双路径"""
            
            # 1. 统计学检测器 (基于MTSExample)
            self.statistical_detector = StatisticalDetector()
            
            # 2. 轻量级深度学习检测器 (专门优化Event连续性)
            self.deep_detector = LightEventDetector(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                window_size=self.window_size
            ).to(self.device)
            
            # 3. 智能投票器 (融合两种方法)
            self.voting_system = IntelligentVotingSystem()
            
        def train_valid_phase(self, tsData):
            """训练阶段: 只训练深度学习组件"""
            print(f"[LOG] VotingEnsemble.train_valid_phase() 调用，数据形状: {tsData.train.shape}")
            
            train_data = tsData.train
            
            # 统计学方法不需要训练
            print("[LOG] 统计学检测器无需训练")
            
            # 训练轻量级深度学习模型
            print("[LOG] 开始训练轻量级深度学习检测器...")
            self._train_deep_detector(train_data)
            print("[LOG] 深度学习检测器训练完成")
            
        def _train_deep_detector(self, train_data):
            """训练轻量级深度学习检测器"""
            
            # 创建训练窗口
            windows = self._create_windows(train_data)
            if len(windows) == 0:
                print("[LOG] 警告: 训练数据不足，跳过深度学习训练")
                return
                
            train_dataset = TensorDataset(torch.FloatTensor(windows))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # 优化器设置 (轻量级训练)
            optimizer = optim.Adam(self.deep_detector.parameters(), lr=1e-3)
            
            # 简单快速训练 (10轮即可)
            self.deep_detector.train()
            for epoch in range(10):
                epoch_loss = 0
                for batch_idx, (batch_data,) in enumerate(train_loader):
                    batch_data = batch_data.to(self.device)
                    
                    # 自监督重构损失
                    loss = self.deep_detector.compute_loss(batch_data)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if epoch % 5 == 0:
                    avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
                    print(f"[LOG] Deep Detector Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            self.deep_detector.eval()
            
        def test_phase(self, tsData: MTSData):
            """测试阶段: 双路径检测 + 智能投票融合"""
            print(f"[LOG] VotingEnsemble.test_phase() 调用，测试数据形状: {tsData.test.shape}")
            
            test_data = tsData.test
            
            # === 路径1: 统计学检测 (保持MTSExample优势) ===
            print("[LOG] 执行统计学检测...")
            stat_scores = self.statistical_detector.detect(test_data)
            
            # === 路径2: 深度学习检测 (专门优化Event连续性) ===
            print("[LOG] 执行深度学习检测...")
            deep_scores = self._deep_detect(test_data)
            
            # === 路径3: 智能投票融合 ===
            print("[LOG] 执行智能投票融合...")
            final_scores = self.voting_system.vote(
                statistical_scores=stat_scores,
                deep_scores=deep_scores,
                original_data=test_data
            )
            
            # 最终归一化
            if len(final_scores) > 0:
                final_scores = (final_scores - np.min(final_scores)) / (np.max(final_scores) - np.min(final_scores) + 1e-10)
            
            self.__anomaly_score = final_scores
            print(f"[LOG] VotingEnsemble异常分数计算完成，长度: {len(final_scores)}")
            print(f"[LOG] 分数统计: min={np.min(final_scores):.4f}, max={np.max(final_scores):.4f}, mean={np.mean(final_scores):.4f}")
            
        def _deep_detect(self, test_data):
            """深度学习检测"""
            windows = self._create_sliding_windows(test_data)
            
            if len(windows) == 0:
                # 如果无法创建窗口，返回统计学分数
                return np.zeros(len(test_data))
            
            self.deep_detector.eval()
            all_scores = []
            
            with torch.no_grad():
                for i in range(0, len(windows), 16):  # 小批量处理
                    batch = windows[i:i+16]
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    
                    # 深度学习异常分数
                    scores = self.deep_detector.predict(batch_tensor)
                    scores_numpy = scores.cpu().numpy()
                    
                    if scores_numpy.ndim == 0:
                        scores_numpy = np.array([scores_numpy])
                    
                    all_scores.extend(scores_numpy)
            
            # 对齐到原始长度
            aligned_scores = self._align_scores_to_original(all_scores, len(test_data))
            return aligned_scores
        
        def _create_windows(self, data):
            """创建训练窗口"""
            if len(data) < self.window_size:
                return np.array([])
            
            windows = []
            for i in range(0, len(data) - self.window_size + 1, self.window_size // 2):
                window = data[i:i + self.window_size]
                windows.append(window)
            return np.array(windows)
        
        def _create_sliding_windows(self, data):
            """创建滑动窗口"""
            if len(data) < self.window_size:
                return np.array([])
                
            windows = []
            for i in range(len(data) - self.window_size + 1):
                window = data[i:i + self.window_size]
                windows.append(window)
            return np.array(windows)
        
        def _align_scores_to_original(self, scores, original_length):
            """将窗口分数对齐到原始时间序列长度"""
            if len(scores) == 0:
                return np.zeros(original_length)
            
            aligned = np.zeros(original_length)
            count = np.zeros(original_length)
            
            for i, score in enumerate(scores):
                start_idx = i
                end_idx = min(i + self.window_size, original_length)
                aligned[start_idx:end_idx] += score
                count[start_idx:end_idx] += 1
            
            mask = count > 0
            aligned[mask] /= count[mask]
            
            return aligned
            
        def anomaly_score(self) -> np.ndarray:
            print(f"[LOG] VotingEnsemble.anomaly_score() 调用")
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            print(f"[LOG] VotingEnsemble.param_statistic() 调用，保存到: {save_file}")
            
            # 计算深度学习模型参数
            deep_params = sum(p.numel() for p in self.deep_detector.parameters())
            
            param_info = f"""
                VotingEnsemble算法参数统计:

                🎯 设计目标: 统计学+深度学习投票融合异常检测
                📊 预期性能: Point F1: 93%+ (保持MTSExample优势), Event F1: 80%+ (深度学习增强)

                🏗️ 双路径架构:
                1. 统计学检测器: MTSExample的L2范数方法 (0参数)
                2. 轻量级深度学习检测器: 专门优化Event连续性 (~{deep_params}参数)
                3. 智能投票系统: 自适应融合两种方法

                🔢 参数统计:
                - 统计学路径: 0个参数 (纯计算)
                - 深度学习路径: ~{deep_params:,}个参数
                - 总参数量: ~{deep_params:,}个 (相比EventMaster大幅减少)
                - 窗口大小: {self.window_size} (轻量级设计)

                💡 核心创新:
                1. 双路径设计: 统计学保证Point精度，深度学习优化Event连续性
                2. 智能投票: 自适应权重，任务导向融合
                3. 轻量级训练: 仅10轮训练，快速收敛
                4. 专门后处理: 针对Event F1优化的连接策略

                🚀 技术优势:
                - 保持MTSExample的Point F1优势 (93%+)
                - 通过深度学习大幅提升Event F1 (目标80%+)
                - 轻量级设计，训练和推理都很快
                - 工程友好，易于部署和调试

                ⚡ 投票策略:
                - Point精度: 统计学主导 (70%) + 深度学习辅助 (30%)
                - Event连续性: 深度学习主导 (60%) + 统计学基础 (40%)
                - 自适应权重: 根据数据特性动态调整

                🎯 设计哲学:
                "取长补短，投票决策" - 统计学的Point精度 + 深度学习的Event连续性
            """
            
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(param_info)


    # ================== 组件定义 ==================

    class StatisticalDetector:
        """统计学检测器: 基于MTSExample的成功方法"""
        
        def detect(self, data):
            """MTSExample的L2范数方法"""
            # 完全复制MTSExample的成功逻辑
            scores = np.sum(np.square(data), axis=1)
            
            if len(scores) > 0:
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                
            return scores


    class LightEventDetector(nn.Module):
        """轻量级深度学习检测器: 专门优化Event连续性"""
        
        def __init__(self, input_dim, hidden_dim, window_size):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.window_size = window_size
            
            # 极简架构: 只有一个LSTM + 一个全连接层
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            )
            
            # 异常分数预测头
            self.score_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            # x: [batch, seq_len, features]
            lstm_out, _ = self.lstm(x)
            # 取最后一个时间步
            last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
            score = self.score_head(last_output)  # [batch, 1]
            return score.squeeze(-1)  # [batch]
        
        def compute_loss(self, x):
            """自监督重构损失"""
            # 简单的自监督任务: 预测窗口内的异常程度
            # 基于窗口内变异度的大小
            batch_size, seq_len, features = x.size()
            
            # 计算真实的变异度作为伪标签
            window_var = torch.var(x.view(batch_size, -1), dim=1)  # [batch]
            window_var = (window_var - window_var.min()) / (window_var.max() - window_var.min() + 1e-8)
            
            # 模型预测
            pred_score = self.forward(x)  # [batch]
            
            # MSE损失
            loss = nn.MSELoss()(pred_score, window_var)
            return loss
        
        def predict(self, x):
            """预测异常分数"""
            return self.forward(x)


    class IntelligentVotingSystem:
        """智能投票系统: 自适应融合统计学和深度学习"""
        
        def __init__(self):
            self.isolation_forest = None
            
        def vote(self, statistical_scores, deep_scores, original_data):
            """智能投票融合"""
            
            # === 第1步: 基础融合 ===
            # Point精度导向: 统计学主导
            point_focused = 0.7 * statistical_scores + 0.3 * deep_scores
            
            # Event连续性导向: 深度学习主导  
            event_focused = 0.4 * statistical_scores + 0.6 * deep_scores
            
            # === 第2步: Event连续性增强 ===
            # 对Event导向的分数进行平滑处理
            event_smoothed = gaussian_filter1d(event_focused, sigma=2.0)
            
            # === 第3步: Isolation Forest增强 ===
            if self.isolation_forest is None:
                self.isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=30  # 轻量级设置
                )
                self.isolation_forest.fit(original_data)
            
            if_scores = self.isolation_forest.decision_function(original_data)
            if_scores = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores) + 1e-10)
            
            # === 第4步: 自适应权重融合 ===
            # 计算两种方法的一致性
            consistency = self._compute_consistency(statistical_scores, deep_scores)
            
            # 一致性高时偏向统计学 (保证Point精度)
            # 一致性低时增加深度学习权重 (可能的新异常)
            adaptive_weight = 0.6 + 0.2 * consistency  # [0.6, 0.8]
            
            # 最终融合
            final_scores = (
                adaptive_weight * point_focused +          # 保证Point精度
                (1 - adaptive_weight) * event_smoothed +   # 增强Event连续性  
                0.1 * if_scores                            # Isolation Forest增强
            )
            
            # === 第5步: Event连接优化 ===
            connected_scores = self._connect_nearby_anomalies(final_scores)
            
            return connected_scores
        
        def _compute_consistency(self, scores1, scores2):
            """计算两种方法的一致性"""
            # 计算分数的相关系数
            correlation = np.corrcoef(scores1, scores2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 转换到[0, 1]范围
            consistency = (correlation + 1) / 2
            return consistency
        
        def _connect_nearby_anomalies(self, scores, gap_threshold=2):
            """连接相近的异常点，专门优化Event F1"""
            threshold = np.percentile(scores, 85)
            anomaly_mask = scores > threshold
            
            # 填充小间隙
            result_mask = anomaly_mask.copy()
            gap_count = 0
            
            for i in range(1, len(anomaly_mask) - 1):
                if not anomaly_mask[i]:
                    gap_count += 1
                else:
                    if gap_count > 0 and gap_count <= gap_threshold:
                        result_mask[i-gap_count:i] = True
                    gap_count = 0
            
            # 应用连接结果
            connected_scores = scores.copy()
            connected_scores[result_mask] = np.maximum(
                connected_scores[result_mask],
                threshold * 0.9
            )
            
            return connected_scores


    print("[LOG] VotingEnsemble类定义完成")
    
    """============= Run VotingEnsemble ============="""
    training_schema = "mts"
    method = "VotingEnsemble"
    
    print(f"[LOG] 开始运行实验，method={method}, training_schema={training_schema}")
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        preprocess="z-score",  # 继承MTSExample的成功经验
    )
    print("[LOG] 实验运行完成")
       
    """============= [EVALUATION SETTINGS] ============="""
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
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
    print("[LOG] 开始绘图")
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
    print("[LOG] 绘图完成")
    
    print("[LOG] VotingEnsemble执行完毕")
    print("=" * 80)
    print("🎯 VotingEnsemble设计理念:")
    print("   '取长补短，投票决策' - 统计学的Point精度 + 深度学习的Event连续性")
    print("   保持MTSExample优势: Point F1: 93%+")
    print("   深度学习增强: Event F1目标: 80%+")
    print("   智能投票: 自适应权重，任务导向融合")
    print("=" * 80) 