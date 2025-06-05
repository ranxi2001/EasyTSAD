#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAD vs LightMMoE 架构图生成器
生成两种算法的详细技术架构图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_cad_architecture():
    """创建CAD架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # 增大图表尺寸
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'process': '#E8F5E8', 
        'gate': '#FFF3E0',
        'expert': '#F3E5F5',
        'output': '#FFEBEE'
    }
    
    # 输入层 - 减小高度，增大字体
    input_box = FancyBboxPatch((1, 10.5), 8, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.9, '输入数据 [B, Window×Features]', 
            ha='center', va='center', fontsize=16, fontweight='bold')  # 增大字体
    
    # 数据展平
    flatten_box = FancyBboxPatch((3.5, 9), 3, 0.6,  # 减小高度
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['process'],
                                 edgecolor='black', linewidth=1)
    ax.add_patch(flatten_box)
    ax.text(5, 9.3, '数据展平 Flatten', ha='center', va='center', fontsize=14)  # 增大字体
    
    # 门控网络
    gate_box = FancyBboxPatch((0.5, 7), 2.2, 1,  # 减小尺寸
                              boxstyle="round,pad=0.1",
                              facecolor=colors['gate'],
                              edgecolor='black', linewidth=1)
    ax.add_patch(gate_box)
    ax.text(1.6, 7.5, '门控网络\nSoftmax Gate', 
            ha='center', va='center', fontsize=13, fontweight='bold')  # 增大字体
    
    # 专家网络 - 减小尺寸
    expert_positions = [(4, 7), (5.5, 7), (7, 7)]
    expert_labels = ['专家1', '专家2', '专家3']
    
    for i, (pos, label) in enumerate(zip(expert_positions, expert_labels)):
        expert_box = FancyBboxPatch(pos, 1, 1,  # 减小尺寸
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['expert'],
                                    edgecolor='black', linewidth=1)
        ax.add_patch(expert_box)
        ax.text(pos[0] + 0.5, pos[1] + 0.5, label,
                ha='center', va='center', fontsize=13, fontweight='bold')  # 增大字体
    
    # 加权融合
    fusion_box = FancyBboxPatch((3.5, 5.2), 3, 0.8,  # 减小高度
                                boxstyle="round,pad=0.1",
                                facecolor=colors['process'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 5.6, '加权融合\nWeighted Sum', 
            ha='center', va='center', fontsize=14, fontweight='bold')  # 增大字体
    
    # 输出层
    output_box = FancyBboxPatch((3.5, 3.5), 3, 0.8,  # 减小高度
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 3.9, '输出层 Linear', 
            ha='center', va='center', fontsize=14, fontweight='bold')  # 增大字体
    
    # 异常分数
    score_box = FancyBboxPatch((3.5, 1.8), 3, 0.8,  # 减小高度
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(score_box)
    ax.text(5, 2.2, '异常分数\n[B, Features]', 
            ha='center', va='center', fontsize=14, fontweight='bold')  # 增大字体
    
    # 绘制连接线 - 调整位置
    connections = [
        # 输入到展平
        ((5, 10.5), (5, 9.6)),
        # 展平到门控和专家
        ((4.5, 9), (2.2, 8)),
        ((5, 9), (4.5, 8)),
        ((5.2, 9), (6, 8)),
        ((5.5, 9), (7.5, 8)),
        # 门控到融合
        ((2.2, 7), (4, 6)),
        # 专家到融合
        ((4.5, 7), (4.5, 6)),
        ((6, 7), (5.5, 6)),
        ((7.5, 7), (6, 6)),
        # 融合到输出
        ((5, 5.2), (5, 4.3)),
        # 输出到分数
        ((5, 3.5), (5, 2.6))
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))  # 增粗箭头
    
    # 添加标题
    ax.text(5, 11.7, 'CAD算法架构图', ha='center', va='center', 
            fontsize=20, fontweight='bold')  # 增大标题字体
    
    # 添加特性说明
    ax.text(0.5, 0.5, '特点：简化架构、快速训练、轻量级专家网络', 
            ha='left', va='center', fontsize=12, style='italic',  # 增大字体
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('ForPPT/CAD_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('ForPPT/CAD_architecture.pdf', bbox_inches='tight')
    print("CAD架构图已保存")

def create_lightmmoe_architecture():
    """创建LightMMoE架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 16))  # 增大图表尺寸
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'conv': '#E8F5E8',
        'expert': '#F3E5F5', 
        'gate': '#FFF3E0',
        'fusion': '#FFEB3B',
        'tower': '#FFE0B2',
        'output': '#FFEBEE'
    }
    
    # 输入层
    input_box = FancyBboxPatch((4, 14.5), 6, 1,  # 减小高度
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 15, '输入数据 [B, 16, 38]', 
            ha='center', va='center', fontsize=16, fontweight='bold')  # 增大字体
    
    # CNN特征提取
    conv_box = FancyBboxPatch((4, 12.8), 6, 1,  # 减小高度
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['conv'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(conv_box)
    ax.text(7, 13.3, 'CNN特征提取\nConv2d(1,8,(16,1))+ReLU+Dropout', 
            ha='center', va='center', fontsize=13, fontweight='bold')  # 增大字体
    
    # 专家网络 (4个) - 减小尺寸
    expert_positions = [(1, 10.5), (3.5, 10.5), (6, 10.5), (8.5, 10.5)]
    expert_labels = ['专家1', '专家2', '专家3', '专家4']
    
    for i, (pos, label) in enumerate(zip(expert_positions, expert_labels)):
        expert_box = FancyBboxPatch(pos, 1.8, 1.2,  # 减小尺寸
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['expert'],
                                    edgecolor='black', linewidth=1)
        ax.add_patch(expert_box)
        ax.text(pos[0] + 0.9, pos[1] + 0.6, label,
                ha='center', va='center', fontsize=13, fontweight='bold')  # 增大字体
    
    # 门控机制 - 减小尺寸
    gate_specific = FancyBboxPatch((11, 12.8), 2.2, 0.8,  # 减小尺寸
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['gate'],
                                   edgecolor='black', linewidth=1)
    ax.add_patch(gate_specific)
    ax.text(12.1, 13.2, '任务特定\n门控', 
            ha='center', va='center', fontsize=12, fontweight='bold')  # 增大字体
    
    gate_shared = FancyBboxPatch((11, 11.5), 2.2, 0.8,  # 减小尺寸
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['gate'],
                                 edgecolor='black', linewidth=1)
    ax.add_patch(gate_shared)
    ax.text(12.1, 11.9, '共享\n门控', 
            ha='center', va='center', fontsize=12, fontweight='bold')  # 增大字体
    
    # 门控融合
    gate_fusion = FancyBboxPatch((11, 9.8), 2.2, 0.8,  # 减小尺寸
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['fusion'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(gate_fusion)
    ax.text(12.1, 10.2, '门控融合\n混合权重', 
            ha='center', va='center', fontsize=12, fontweight='bold')  # 增大字体
    
    # 加权专家融合
    expert_fusion = FancyBboxPatch((4, 8), 6, 1.2,  # 减小高度
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['fusion'],
                                   edgecolor='black', linewidth=2)
    ax.add_patch(expert_fusion)
    ax.text(7, 8.6, '加权专家融合\ngate_weights × expert_outputs', 
            ha='center', va='center', fontsize=13, fontweight='bold')  # 增大字体
    
    # 塔网络 (显示3个代表38个) - 减小尺寸
    tower_positions = [(2, 5.5), (6, 5.5), (10, 5.5)]
    tower_labels = ['Tower 1', 'Tower ...', 'Tower 38']
    
    for i, (pos, label) in enumerate(zip(tower_positions, tower_labels)):
        tower_box = FancyBboxPatch(pos, 1.8, 1.2,  # 减小尺寸
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['tower'],
                                   edgecolor='black', linewidth=1)
        ax.add_patch(tower_box)
        ax.text(pos[0] + 0.9, pos[1] + 0.6, label,
                ha='center', va='center', fontsize=13, fontweight='bold')  # 增大字体
    
    # 输出
    output_box = FancyBboxPatch((4, 3.2), 6, 1,  # 减小高度
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 3.7, '异常分数 [B, 1, 38]', 
            ha='center', va='center', fontsize=16, fontweight='bold')  # 增大字体
    
    # 绘制连接线 - 调整位置并增粗
    # 输入到CNN
    ax.annotate('', xy=(7, 12.8), xytext=(7, 14.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='blue'))
    
    # CNN到专家网络
    for pos in expert_positions:
        ax.annotate('', xy=(pos[0] + 0.9, pos[1] + 1.2), xytext=(7, 12.8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # 输入到门控
    ax.annotate('', xy=(11, 13.2), xytext=(10, 15),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(11, 11.9), xytext=(10, 15),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    # 门控到融合
    ax.annotate('', xy=(12.1, 9.8), xytext=(12.1, 11.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    # 专家到融合
    for pos in expert_positions:
        ax.annotate('', xy=(pos[0] + 0.9, 9.2), xytext=(pos[0] + 0.9, pos[1]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    
    # 门控融合到专家融合
    ax.annotate('', xy=(10, 8.6), xytext=(11, 10.2),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
    
    # 专家融合到塔网络
    for pos in tower_positions:
        ax.annotate('', xy=(pos[0] + 0.9, pos[1] + 1.2), xytext=(7, 8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='brown'))
    
    # 塔网络到输出
    for pos in tower_positions:
        ax.annotate('', xy=(7, 4.2), xytext=(pos[0] + 0.9, pos[1]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # 添加标题
    ax.text(7, 15.8, 'LightMMoE算法架构图', ha='center', va='center',
            fontsize=20, fontweight='bold')  # 增大标题字体
    
    # 添加特性说明
    ax.text(0.5, 1, '特点：卷积特征提取、混合门控机制、多任务塔网络、高精度检测', 
            ha='left', va='center', fontsize=12, style='italic',  # 增大字体
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('ForPPT/LightMMoE_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('ForPPT/LightMMoE_architecture.pdf', bbox_inches='tight')
    print("LightMMoE架构图已保存")

def create_comparison_chart():
    """创建架构对比图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))  # 增大图表尺寸
    
    # CAD特征雷达图
    categories = ['参数数量', '训练速度', '推理速度', '内存使用', '实现复杂度', '可扩展性']
    cad_scores = [3, 9, 9, 9, 8, 6]
    lightmmoe_scores = [7, 3, 6, 6, 4, 9]
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    cad_scores += cad_scores[:1]
    lightmmoe_scores += lightmmoe_scores[:1]
    
    ax1.plot(angles, cad_scores, 'o-', linewidth=3, label='CAD', color='blue', markersize=8)
    ax1.fill(angles, cad_scores, alpha=0.25, color='blue')
    ax1.plot(angles, lightmmoe_scores, 'o-', linewidth=3, label='LightMMoE', color='red', markersize=8)
    ax1.fill(angles, lightmmoe_scores, alpha=0.25, color='red')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=13)  # 增大字体
    ax1.set_ylim(0, 10)
    ax1.set_title('架构特性对比雷达图', fontsize=16, fontweight='bold', pad=20)  # 增大字体
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=12)  # 增大字体
    ax1.grid(True)
    
    # 设置雷达图的刻度标签字体大小
    ax1.tick_params(axis='y', labelsize=11)
    
    # 性能效率散点图
    methods = ['CAD', 'LightMMoE']
    performance = [0.930, 0.934]
    efficiency = [0.0062, 0.00016]
    colors_scatter = ['blue', 'red']
    sizes = [300, 300]  # 增大散点
    
    ax2.scatter(efficiency, performance, c=colors_scatter, s=sizes, alpha=0.7)
    
    for i, method in enumerate(methods):
        ax2.annotate(method, (efficiency[i], performance[i]),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=14, fontweight='bold')  # 增大字体
    
    ax2.set_xlabel('效率比值 (F1/训练时间)', fontsize=14)  # 增大字体
    ax2.set_ylabel('F1性能', fontsize=14)  # 增大字体
    ax2.set_title('性能-效率散点图', fontsize=16, fontweight='bold')  # 增大字体
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=12)  # 增大刻度字体
    
    # 添加象限标签
    ax2.axhline(y=np.mean(performance), color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=np.mean(efficiency), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ForPPT/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('ForPPT/architecture_comparison.pdf', bbox_inches='tight')
    print("架构对比图表已保存")

def main():
    """主函数"""
    print("开始生成CAD vs LightMMoE架构图...")
    
    # 创建目录
    import os
    os.makedirs('ForPPT', exist_ok=True)
    
    # 生成架构图
    create_cad_architecture()
    create_lightmmoe_architecture() 
    create_comparison_chart()
    
    print("所有架构图生成完成!")
    print("文件保存在ForPPT目录下:")
    print("- CAD_architecture.png/pdf")
    print("- LightMMoE_architecture.png/pdf") 
    print("- architecture_comparison.png/pdf")

if __name__ == "__main__":
    main() 