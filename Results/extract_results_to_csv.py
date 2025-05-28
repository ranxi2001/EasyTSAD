#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果提取脚本
遍历Results/Evals目录下所有方法的评估结果，并生成CSV表格
"""

import os
import json
import pandas as pd
from pathlib import Path

def extract_results_to_csv():
    """
    从Results/Evals目录提取所有方法的评估结果，生成CSV文件
    """
    # 设置路径 - 修复路径问题
    script_dir = Path(__file__).parent  # 获取脚本所在目录
    evals_dir = script_dir / "Evals"    # Results/Evals
    output_file = script_dir / "evaluation_results.csv"
    
    print(f"当前工作目录: {Path.cwd()}")
    print(f"脚本目录: {script_dir}")
    print(f"Evals目录: {evals_dir}")
    print(f"Evals目录是否存在: {evals_dir.exists()}")
    
    # 存储所有结果的列表
    results_data = []
    
    print("开始提取评估结果...")
    
    if not evals_dir.exists():
        print(f"错误: 找不到目录 {evals_dir}")
        return
    
    # 遍历所有方法文件夹
    for method_dir in evals_dir.iterdir():
        if not method_dir.is_dir():
            continue
            
        method_name = method_dir.name
        print(f"处理方法: {method_name}")
        
        # 遍历训练模式（通常是mts）
        for schema_dir in method_dir.iterdir():
            if not schema_dir.is_dir():
                continue
                
            schema_name = schema_dir.name
            
            # 遍历数据集文件夹
            for dataset_dir in schema_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                    
                dataset_name = dataset_dir.name
                
                # 查找avg.json文件
                avg_json_path = dataset_dir / "avg.json"
                if avg_json_path.exists():
                    try:
                        # 读取JSON文件
                        with open(avg_json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 提取每种评估指标的结果
                        for metric_name, metric_data in data.items():
                            if isinstance(metric_data, dict) and 'f1' in metric_data:
                                result_row = {
                                    'Method': method_name,
                                    'Schema': schema_name,
                                    'Dataset': dataset_name,
                                    'Metric': metric_name,
                                    'F1': metric_data.get('f1', 0),
                                    'Precision': metric_data.get('precision', 0),
                                    'Recall': metric_data.get('recall', 0)
                                }
                                results_data.append(result_row)
                                
                        print(f"  - 已处理: {dataset_name}")
                        
                    except Exception as e:
                        print(f"  - 错误处理 {avg_json_path}: {e}")
                else:
                    print(f"  - 未找到 avg.json: {dataset_dir}")
    
    # 创建DataFrame并保存为CSV
    if results_data:
        df = pd.DataFrame(results_data)
        
        # 重新排序列
        column_order = ['Method', 'Schema', 'Dataset', 'Metric', 'F1', 'Precision', 'Recall']
        df = df[column_order]
        
        # 按方法、数据集、指标排序
        df = df.sort_values(['Method', 'Dataset', 'Metric'])
        
        # 保存为CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        print(f"总共提取了 {len(results_data)} 条记录")
        
        # 显示数据预览
        print("\n数据预览:")
        print(df.head(10))
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"方法数量: {df['Method'].nunique()}")
        print(f"数据集数量: {df['Dataset'].nunique()}")
        print(f"指标类型: {df['Metric'].nunique()}")
        print(f"方法列表: {', '.join(df['Method'].unique())}")
        print(f"数据集列表: {', '.join(df['Dataset'].unique())}")
        
    else:
        print("未找到任何有效的评估结果")

def create_pivot_tables():
    """
    创建透视表，便于对比不同方法的性能
    """
    script_dir = Path(__file__).parent
    csv_file = script_dir / "evaluation_results.csv"
    
    if not csv_file.exists():
        print(f"错误: 未找到 {csv_file} 文件")
        return
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 为每种指标类型创建透视表
    metrics = df['Metric'].unique()
    
    for metric in metrics:
        metric_data = df[df['Metric'] == metric]
        
        # 创建F1分数的透视表
        f1_pivot = metric_data.pivot_table(
            values='F1', 
            index='Method', 
            columns='Dataset', 
            aggfunc='mean'
        )
        
        # 保存透视表
        output_file = script_dir / f"pivot_f1_{metric.replace(' ', '_').replace('/', '_')}.csv"
        f1_pivot.to_csv(output_file, encoding='utf-8-sig')
        print(f"F1透视表已保存: {output_file}")
        
        # 创建Precision的透视表
        precision_pivot = metric_data.pivot_table(
            values='Precision', 
            index='Method', 
            columns='Dataset', 
            aggfunc='mean'
        )
        
        output_file = script_dir / f"pivot_precision_{metric.replace(' ', '_').replace('/', '_')}.csv"
        precision_pivot.to_csv(output_file, encoding='utf-8-sig')
        print(f"Precision透视表已保存: {output_file}")
        
        # 创建Recall的透视表
        recall_pivot = metric_data.pivot_table(
            values='Recall', 
            index='Method', 
            columns='Dataset', 
            aggfunc='mean'
        )
        
        output_file = script_dir / f"pivot_recall_{metric.replace(' ', '_').replace('/', '_')}.csv"
        recall_pivot.to_csv(output_file, encoding='utf-8-sig')
        print(f"Recall透视表已保存: {output_file}")

def create_summary_table():
    """
    创建汇总表，显示每个方法的最佳性能
    """
    script_dir = Path(__file__).parent
    csv_file = script_dir / "evaluation_results.csv"
    
    if not csv_file.exists():
        print(f"错误: 未找到 {csv_file} 文件")
        return
    
    df = pd.read_csv(csv_file)
    
    # 计算每个方法在每个指标上的平均性能
    summary = df.groupby(['Method', 'Metric']).agg({
        'F1': ['mean', 'std', 'max'],
        'Precision': ['mean', 'std', 'max'],
        'Recall': ['mean', 'std', 'max']
    }).round(4)
    
    # 展平列名
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # 保存汇总表
    output_file = script_dir / "summary_results.csv"
    summary.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"性能汇总表已保存: {output_file}")
    
    # 找出每个指标的最佳方法
    print("\n各指标最佳方法:")
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        best_method = metric_data.loc[metric_data['F1'].idxmax()]
        print(f"{metric}:")
        print(f"  最佳方法: {best_method['Method']}")
        print(f"  数据集: {best_method['Dataset']}")
        print(f"  F1分数: {best_method['F1']:.4f}")
        print()

if __name__ == "__main__":
    print("=" * 60)
    print("评估结果提取工具")
    print("=" * 60)
    
    # 提取结果到CSV
    extract_results_to_csv()
    
    print("\n" + "=" * 60)
    print("创建透视表...")
    print("=" * 60)
    
    # 创建透视表
    create_pivot_tables()
    
    print("\n" + "=" * 60)
    print("创建汇总表...")
    print("=" * 60)
    
    # 创建汇总表
    create_summary_table()
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60) 