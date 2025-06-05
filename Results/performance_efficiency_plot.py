#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能-效率象限散点图生成器
读取运行时间和平均best F1分数，创建性能效率分析图表
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置字体和样式（使用英文避免字体问题）
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def read_runtime_data():
    """读取所有方法的运行时间数据"""
    runtime_data = {}
    
    # 确定脚本运行路径
    script_dir = Path(__file__).parent  # Results目录
    runtime_path = script_dir / "RunTime"
    
    print(f"脚本目录: {script_dir}")
    print(f"查找RunTime目录: {runtime_path}")
    
    if not runtime_path.exists():
        print(f"警告：找不到RunTime目录：{runtime_path.absolute()}")
        return runtime_data
    
    for method_dir in runtime_path.iterdir():
        if method_dir.is_dir():
            method_name = method_dir.name
            # 查找 machine-1 数据集的时间数据
            time_file = method_dir / "mts" / "machine-1" / "time.json"
            if time_file.exists():
                try:
                    with open(time_file, 'r') as f:
                        time_data = json.load(f)
                        # 获取训练+验证时间
                        train_time = time_data.get("train_and_valid", 0)
                        runtime_data[method_name] = train_time
                        print(f"成功读取 {method_name}: {train_time:.4f} 秒")
                except Exception as e:
                    print(f"读取 {method_name} 时间数据时出错: {e}")
            else:
                print(f"找不到时间文件: {time_file}")
    
    print(f"总共读取到 {len(runtime_data)} 个方法的运行时间数据")
    return runtime_data

def read_performance_data():
    """读取所有方法的best F1性能数据"""
    performance_data = {}
    
    # 确定脚本运行路径
    script_dir = Path(__file__).parent  # Results目录
    summary_file = script_dir / "summary_results.csv"
    
    print(f"查找性能数据文件: {summary_file}")
    
    if not summary_file.exists():
        print(f"警告：找不到性能数据文件：{summary_file.absolute()}")
        return performance_data
        
    print(f"读取性能数据文件：{summary_file.absolute()}")
    df = pd.read_csv(summary_file)
    # 筛选出best f1 under pa指标的数据
    best_f1_data = df[df['Metric'] == 'best f1 under pa']
    
    for _, row in best_f1_data.iterrows():
        method_name = row['Method']
        f1_mean = row['F1_mean']
        performance_data[method_name] = f1_mean
        print(f"成功读取 {method_name}: F1={f1_mean:.4f}")
    
    print(f"总共读取到 {len(performance_data)} 个方法的性能数据")
    return performance_data

def categorize_complexity(train_time):
    """根据训练时间对复杂度进行分类"""
    if train_time < 1:
        return "Very Low", "#2E8B57"  # 海绿色
    elif train_time < 60:
        return "Low", "#4169E1"   # 皇家蓝
    elif train_time < 600:
        return "Medium", "#FFD700"   # 金色
    elif train_time < 6000:
        return "High", "#FF6347"   # 番茄红
    else:
        return "Very High", "#8B0000"  # 暗红色

def create_performance_efficiency_plot():
    """创建性能-效率象限散点图"""
    # 读取数据
    runtime_data = read_runtime_data()
    performance_data = read_performance_data()
    
    print(f"\n数据匹配检查:")
    print(f"运行时间数据: {len(runtime_data)} 个方法")
    print(f"性能数据: {len(performance_data)} 个方法")
    
    # 合并数据
    methods = []
    train_times = []
    f1_scores = []
    complexities = []
    colors = []
    
    for method in runtime_data.keys():
        if method in performance_data:
            methods.append(method)
            train_time = runtime_data[method]
            train_times.append(train_time)
            f1_scores.append(performance_data[method])
            
            complexity, color = categorize_complexity(train_time)
            complexities.append(complexity)
            colors.append(color)
            print(f"匹配方法: {method}")
    
    print(f"\n最终匹配到 {len(methods)} 个方法")
    
    # 检查是否有数据
    if len(methods) == 0:
        print("❌ 没有找到匹配的数据，请检查数据文件路径和内容")
        return
    
    # 创建图表
    plt.figure(figsize=(14, 10))
    
    # 绘制散点图
    scatter = plt.scatter(train_times, f1_scores, 
                         c=colors, s=120, alpha=0.7, edgecolors='black', linewidth=1)
    
    # 添加方法名标签
    for i, method in enumerate(methods):
        plt.annotate(method, (train_times[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 设置坐标轴（使用英文）
    plt.xlabel('Training Time (seconds, log scale)', fontsize=14, fontweight='bold')
    plt.ylabel('Performance (Best F1 Score)', fontsize=14, fontweight='bold')
    plt.title('Time Series Anomaly Detection: Performance-Efficiency Analysis', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 使用对数刻度显示训练时间
    plt.xscale('log')
    
    # 设置网格
    plt.grid(True, alpha=0.3)
    
    # 添加象限分割线
    median_time = np.median(train_times)
    median_f1 = np.median(f1_scores)
    
    plt.axvline(x=median_time, color='red', linestyle='--', alpha=0.5, linewidth=2)
    plt.axhline(y=median_f1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # 添加象限标签（使用英文，调整位置避免遮挡数据点）
    xlim = plt.xlim()
    ylim = plt.ylim()
    
    # 右上角 - 高性能高复杂度
    plt.text(xlim[1]*0.8, ylim[1]*0.96, 'High Performance\nHigh Complexity', 
             ha='center', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))
    
    # 左上角 - 高性能低复杂度
    plt.text(xlim[0]*5, ylim[1]*0.96, 'High Performance\nLow Complexity', 
             ha='center', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))
    
    # 右下角 - 低性能高复杂度
    plt.text(xlim[1]*0.8, ylim[0]*1.04, 'Low Performance\nHigh Complexity', 
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8))
    
    # 左下角 - 低性能低复杂度
    plt.text(xlim[0]*5, ylim[0]*1.04, 'Low Performance\nLow Complexity', 
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    
    # 创建复杂度图例（使用英文）
    complexity_levels = ["Very Low", "Low", "Medium", "High", "Very High"]
    complexity_colors = ["#2E8B57", "#4169E1", "#FFD700", "#FF6347", "#8B0000"]
    
    legend_elements = []
    for level, color in zip(complexity_levels, complexity_colors):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=f'{level} Complexity'))
    
    plt.legend(handles=legend_elements, title='Model Complexity', 
              loc='lower left', fontsize=10, title_fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    script_dir = Path(__file__).parent
    png_file = script_dir / 'performance_efficiency_quadrant.png'
    pdf_file = script_dir / 'performance_efficiency_quadrant.pdf'
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print("Performance-Efficiency quadrant plot saved as:")
    print(f"- {png_file.absolute()}")
    print(f"- {pdf_file.absolute()}")
    
    # 关闭图表以释放内存
    plt.close()
    
    # 打印统计信息
    print("\n=== Performance-Efficiency Analysis Statistics ===")
    print(f"Total methods evaluated: {len(methods)}")
    if len(train_times) > 0:
        print(f"Training time range: {min(train_times):.4f} - {max(train_times):.2f} seconds")
        print(f"F1 score range: {min(f1_scores):.4f} - {max(f1_scores):.4f}")
        print(f"Median training time: {median_time:.2f} seconds")
        print(f"Median F1 score: {median_f1:.4f}")
        
        # 识别帕累托前沿（高性能低复杂度的方法）
        print("\n=== Pareto Optimal Methods (High Performance Low Complexity) ===")
        pareto_methods = []
        for i, method in enumerate(methods):
            if f1_scores[i] > median_f1 and train_times[i] < median_time:
                pareto_methods.append(method)
                print(f"- {method}: F1={f1_scores[i]:.4f}, Training time={train_times[i]:.4f}s")
        
        if not pareto_methods:
            print("No methods found in the high performance low complexity quadrant")

def create_detailed_comparison_table():
    """创建详细的性能-效率对比表"""
    runtime_data = read_runtime_data()
    performance_data = read_performance_data()
    
    # 合并数据并创建DataFrame
    comparison_data = []
    for method in runtime_data.keys():
        if method in performance_data:
            train_time = runtime_data[method]
            f1_score = performance_data[method]
            complexity, _ = categorize_complexity(train_time)
            
            # 计算效率比率 (性能/时间)
            efficiency_ratio = f1_score / max(train_time, 0.001)  # 避免除零
            
            comparison_data.append({
                'Method': method,
                'F1_Score': f1_score,
                'Train_Time_sec': train_time,
                'Complexity': complexity,
                'Efficiency_Ratio': efficiency_ratio
            })
    
    if not comparison_data:
        print("❌ 没有数据可以创建对比表")
        return
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Efficiency_Ratio', ascending=False)
    
    # 保存详细对比表
    script_dir = Path(__file__).parent
    csv_file = script_dir / 'performance_efficiency_comparison.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\nDetailed comparison table saved as: {csv_file.absolute()}")
    
    # 显示前10名最高效率的方法
    print("\n=== Efficiency Ratio Ranking Top 10 ===")
    print(df.head(10).to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    print("Starting performance-efficiency quadrant plot generation...")
    try:
        create_performance_efficiency_plot()
        create_detailed_comparison_table()
        print("\n✅ Analysis completed!")
    except Exception as e:
        print(f"❌ Error generating plot: {e}")
        import traceback
        traceback.print_exc() 