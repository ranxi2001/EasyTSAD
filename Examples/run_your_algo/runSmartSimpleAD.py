from typing import Dict
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    print("[LOG] 开始运行SmartSimpleAD - 智能简单异常检测算法")
    
    # Create a global controller
    gctrl = TSADController()
    print("[LOG] TSADController已创建")
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["machine-1", "machine-2", "machine-3"]
    dataset_types = "MTS"
    
    print("[LOG] 开始设置数据集")
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    """============= 智能简单异常检测算法 ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    print("[LOG] 开始定义SmartSimpleAD类")
    
    class SmartSimpleAD(BaseMethod):
        """
        SmartSimpleAD: 智能简单异常检测算法
        
        核心哲学: "简单优于复杂，智能胜过粗暴"
        
        设计原理:
        1. 保持L2范数的简单有效性
        2. 添加特征重要性自适应权重
        3. 多时间尺度信息融合
        4. 智能后处理优化
        
        预期效果: 在保持极简实现的同时，获得比MTSExample更好的性能
        """
        
        def __init__(self, params: dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.feature_weights = None
            self.multi_scale_scores = []
            print("[LOG] SmartSimpleAD.__init__() 调用")
            
        def train_valid_phase(self, tsData):
            """
            训练阶段: 计算特征重要性权重
            核心思想: 方差大的特征更重要
            """
            print(f"[LOG] SmartSimpleAD.train_valid_phase() 调用，数据形状: {tsData.train.shape}")
            
            # 计算每个特征的方差作为重要性权重
            feature_vars = np.var(tsData.train, axis=0)
            
            # 归一化权重，避免某个特征过于突出
            self.feature_weights = feature_vars / (np.sum(feature_vars) + 1e-10)
            
            # 平滑权重，避免过度偏向某些特征
            self.feature_weights = np.sqrt(self.feature_weights)
            self.feature_weights = self.feature_weights / (np.sum(self.feature_weights) + 1e-10)
            
            print(f"[LOG] 特征权重计算完成，权重范围: [{np.min(self.feature_weights):.4f}, {np.max(self.feature_weights):.4f}]")
            
        def test_phase(self, tsData: MTSData):
            """
            测试阶段: 多尺度智能异常分数计算
            """
            print(f"[LOG] SmartSimpleAD.test_phase() 调用，测试数据形状: {tsData.test.shape}")
            test_data = tsData.test
            
            # === 核心算法1: 加权L2范数 ===
            # 基于MTSExample的成功，但添加特征重要性权重
            weighted_data = test_data * self.feature_weights  # 广播权重
            base_scores = np.sum(np.square(weighted_data), axis=1)
            
            # === 核心算法2: 多时间尺度检测 ===
            multi_scale_scores = []
            
            # 尺度1: 原始点级别异常（如MTSExample）
            scale1_scores = base_scores
            multi_scale_scores.append(scale1_scores)
            
            # 尺度2: 短期滑动窗口异常（窗口=3）
            if len(test_data) >= 3:
                window_size = 3
                scale2_scores = []
                for i in range(len(test_data)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(test_data), i + window_size // 2 + 1)
                    window_data = test_data[start_idx:end_idx]
                    
                    # 计算窗口内的变异度
                    window_weighted = window_data * self.feature_weights
                    window_var = np.var(window_weighted, axis=0)
                    window_score = np.sum(window_var)
                    scale2_scores.append(window_score)
                
                scale2_scores = np.array(scale2_scores)
                multi_scale_scores.append(scale2_scores)
            
            # 尺度3: 中期趋势异常（窗口=7）
            if len(test_data) >= 7:
                window_size = 7
                scale3_scores = []
                for i in range(len(test_data)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(test_data), i + window_size // 2 + 1)
                    window_data = test_data[start_idx:end_idx]
                    
                    # 计算趋势变化
                    if len(window_data) >= 3:
                        weighted_window = window_data * self.feature_weights
                        # 简单的趋势检测：计算一阶差分的方差
                        diff_data = np.diff(weighted_window, axis=0)
                        trend_score = np.sum(np.var(diff_data, axis=0))
                        scale3_scores.append(trend_score)
                    else:
                        scale3_scores.append(0.0)
                
                scale3_scores = np.array(scale3_scores)
                multi_scale_scores.append(scale3_scores)
            
            # === 核心算法3: 智能融合 ===
            # 自适应权重融合多个尺度
            if len(multi_scale_scores) == 1:
                fused_scores = multi_scale_scores[0]
            else:
                # 归一化各尺度分数
                normalized_scores = []
                for scores in multi_scale_scores:
                    if np.max(scores) > np.min(scores):
                        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                    else:
                        norm_scores = scores
                    normalized_scores.append(norm_scores)
                
                # 智能权重: 点级别权重高，其他尺度作为补充
                if len(normalized_scores) == 2:
                    weights = [0.7, 0.3]  # 点级别70%，短期30%
                elif len(normalized_scores) == 3:
                    weights = [0.6, 0.25, 0.15]  # 点级别60%，短期25%，中期15%
                else:
                    weights = [1.0 / len(normalized_scores)] * len(normalized_scores)
                
                # 加权融合
                fused_scores = np.zeros_like(normalized_scores[0])
                for i, (scores, weight) in enumerate(zip(normalized_scores, weights)):
                    fused_scores += weight * scores
            
            # === 核心算法4: 智能后处理 ===
            # 轻微平滑，增强连续异常的检测效果（适配Point Adjustment评估）
            if len(fused_scores) > 5:
                # 使用高斯滤波进行轻微平滑
                sigma = 0.8  # 较小的sigma，保持大部分细节
                smoothed_scores = gaussian_filter1d(fused_scores, sigma=sigma)
                
                # 自适应混合原始分数和平滑分数
                alpha = 0.85  # 85%原始分数 + 15%平滑分数
                fused_scores = alpha * fused_scores + (1 - alpha) * smoothed_scores
            
            # === 最终归一化 ===
            if len(fused_scores) > 0:
                final_scores = (fused_scores - np.min(fused_scores)) / (np.max(fused_scores) - np.min(fused_scores) + 1e-10)
            else:
                final_scores = fused_scores
                
            self.__anomaly_score = final_scores
            print(f"[LOG] SmartSimpleAD异常分数计算完成，长度: {len(final_scores)}")
            print(f"[LOG] 分数统计: min={np.min(final_scores):.4f}, max={np.max(final_scores):.4f}, mean={np.mean(final_scores):.4f}")
            
        def anomaly_score(self) -> np.ndarray:
            print(f"[LOG] SmartSimpleAD.anomaly_score() 调用")
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            print(f"[LOG] SmartSimpleAD.param_statistic() 调用，保存到: {save_file}")
            param_info = """
                SmartSimpleAD算法参数统计:

                算法复杂度: ⭐ (极简)
                核心理念: 简单优于复杂，智能胜过粗暴

                核心组件:
                1. 特征重要性权重: 38维权重向量
                2. 多尺度检测器: 3个时间尺度
                3. 智能融合器: 自适应权重融合
                4. 智能后处理: 高斯平滑优化

                总参数量: ~150个 (主要是权重向量)
                计算复杂度: O(n*m) where n=时间点数, m=特征数
                内存复杂度: O(n*m)

                设计哲学:
                - 基于MTSExample的L2范数成功经验
                - 添加最小必要的智能机制
                - 保持算法的可解释性和稳定性
                - 适配Point Adjustment评估特性

                预期优势:
                - 比MTSExample更智能的特征权重
                - 多时间尺度提供更全面的异常检测
                - 智能后处理优化评估表现
                - 仍然保持极简的算法复杂度
            """
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(param_info)
    
    print("[LOG] SmartSimpleAD类定义完成")
    
    """============= Run SmartSimpleAD ============="""
    training_schema = "mts"
    method = "SmartSimpleAD"
    
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
    
    print("[LOG] SmartSimpleAD执行完毕")
    print("=" * 80)
    print("🎯 SmartSimpleAD算法设计理念:")
    print("   '简单优于复杂，智能胜过粗暴'")
    print("   基于MTSExample成功经验，添加最小必要的智能机制")
    print("   预期目标: 在保持极简的同时，超越MTSExample的93.66% Point F1")
    print("=" * 80) 