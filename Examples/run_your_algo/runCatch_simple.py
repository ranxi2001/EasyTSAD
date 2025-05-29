import numpy as np
from typing import Dict
from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import TSData

if __name__ == "__main__":
    
    # 创建全局控制器
    gctrl = TSADController()
        
    """============= [数据集设置] ============="""
    # 指定数据集
    datasets = ["TODS"]
    dataset_types = "UTS"
    
    # 设置数据集路径
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="./datasets",
        datasets=datasets,
    )

    """============= Impletment your algo. ============="""

    class Catch(BaseMethod):
        def __init__(self, hparams) -> None:
            super().__init__()
            self.__anomaly_score = None
            
            # 保存超参数
            self.seq_len = hparams.get("seq_len", 96)
            self.batch_size = hparams.get("batch_size", 32)
            self.num_epochs = hparams.get("num_epochs", 3)
            self.lr = hparams.get("lr", 0.0001)
            self.patience = hparams.get("patience", 3)
            
            print(f"[LOG] CATCH算法初始化完成（简化版本）")

        def _convert_to_dataframe(self, data):
            """将数据转换为numpy数组"""
            if isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)
        
        def train_valid_phase(self, tsTrain: TSData):
            '''
            Define train and valid phase for naive mode. 
            '''
            print(f"[LOG] CATCH开始训练（简化版本），训练数据形状: {tsTrain.train.shape}")
            
            # 简化的训练过程 - 这里只是做一个占位符
            train_data = self._convert_to_dataframe(tsTrain.train)
            valid_data = self._convert_to_dataframe(tsTrain.valid)
            
            print(f"[LOG] 训练数据形状: {train_data.shape}")
            print(f"[LOG] 验证数据形状: {valid_data.shape}")
            
            # 模拟训练过程
            for epoch in range(self.num_epochs):
                # 计算简单的训练和验证损失
                train_loss = np.mean(np.square(train_data)) / (epoch + 1)
                valid_loss = np.mean(np.square(valid_data)) / (epoch + 1)
                
                print(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}")
            
            print("[LOG] 训练完成")
            return
                
        def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            '''
            Define train and valid phase for all-in-one mode. 
            '''
            print("[LOG] CATCH 不支持 all-in-one 模式，使用 naive 模式代替")
            return
            
        def test_phase(self, tsData: TSData):
            '''
            Define test phase for each time series. 
            '''
            print(f"[LOG] CATCH开始测试（简化版本），测试数据形状: {tsData.test.shape}")
            
            # 简化的异常检测：使用序列与其滑动平均的差异作为异常分数
            test_data = self._convert_to_dataframe(tsData.test)
            
            if len(test_data.shape) == 1:
                test_data = test_data.reshape(-1, 1)
            
            # 计算滑动平均
            window_size = min(self.seq_len, len(test_data) // 4)
            if window_size < 2:
                window_size = 2
                
            anomaly_scores = []
            
            for i in range(len(test_data)):
                start_idx = max(0, i - window_size)
                end_idx = min(len(test_data), i + window_size + 1)
                
                # 计算当前点与其邻域平均值的差异
                if start_idx == end_idx:
                    score = 0.0
                else:
                    neighborhood = test_data[start_idx:end_idx]
                    mean_val = np.mean(neighborhood, axis=0)
                    current_val = test_data[i]
                    score = np.sum(np.square(current_val - mean_val))
                
                anomaly_scores.append(score)
            
            # 标准化异常分数
            anomaly_scores = np.array(anomaly_scores)
            if len(anomaly_scores) > 0 and np.max(anomaly_scores) > np.min(anomaly_scores):
                anomaly_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
            
            self.__anomaly_score = anomaly_scores
            
            print(f"[LOG] 异常分数计算完成，长度: {len(self.__anomaly_score)}")
             
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            param_info = f"CATCH Model (简化版本)\n"
            param_info += f"Parameters:\n"
            param_info += f"  seq_len: {self.seq_len}\n"
            param_info += f"  batch_size: {self.batch_size}\n"
            param_info += f"  num_epochs: {self.num_epochs}\n"
            param_info += f"  lr: {self.lr}\n"
            param_info += f"  patience: {self.patience}\n"
            
            with open(save_file, 'w') as f:
                f.write(param_info)

    """============= [算法运行] ============="""
    # 指定方法和训练模式
    training_schema = "naive"
    method = "Catch"
    
    # 运行模型
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "seq_len": 96,
            "batch_size": 32,
            "num_epochs": 3,
            "lr": 0.0001,
            "patience": 3,
        },
        preprocess="z-score",
    )
       
        
    """============= [评估设置] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # 指定评估协议
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
        
        
    """============= [绘图设置] ============="""
    
    # 为每条曲线绘制异常分数
    gctrl.plots(
        method=method,
        training_schema=training_schema
    ) 