from typing import Dict
import numpy as np
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    print("[LOG] 开始运行runMTS.py")
    
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

    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    print("[LOG] 开始定义MTSExample类")
    class MTSExample(BaseMethod):  #这里不用继承BaseMethod也可以，但继承后可以使用EasyTSAD的接口
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            print("[LOG] MTSExample.__init__() 调用")
            
        def train_valid_phase(self, tsData):
            print(f"[LOG] MTSExample.train_valid_phase() 调用，数据形状: {tsData.train.shape}")
            pass
            
        def test_phase(self, tsData: MTSData):
            print(f"[LOG] MTSExample.test_phase() 调用，测试数据形状: {tsData.test.shape}")
            test_data = tsData.test
            
            scores = np.sum(np.square(test_data), axis=1)
            
            if len(scores) > 0:
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                
            self.__anomaly_score = scores
            print(f"[LOG] 异常分数计算完成，长度: {len(scores)}")
            
        def anomaly_score(self) -> np.ndarray:
            print(f"[LOG] MTSExample.anomaly_score() 调用")
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            print(f"[LOG] MTSExample.param_statistic() 调用，保存到: {save_file}")
            param_info = "Your Algo. info"
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    print("[LOG] MTSExample类定义完成")
    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas

    training_schema = "mts"
    method = "MTSExample"  # string of your algo class
    
    print(f"[LOG] 开始运行实验，method={method}, training_schema={training_schema}")
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        # hparams={
        #     "param_1": 2,
        # },
        # use which method to preprocess original data. 
        # Default: raw
        # Option: 
        #   - z-score(Standardlization), 
        #   - min-max(Normalization), 
        #   - raw (original curves)
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
    
    print("[LOG] runMTS.py执行完毕")
