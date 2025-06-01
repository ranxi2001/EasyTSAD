#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成requirements.txt的简化版本
"""

import subprocess
import sys
import os
from datetime import datetime

def generate_requirements_simple():
    """快速生成requirements.txt文件"""
    print("🚀 快速生成requirements.txt...")
    
    # 获取当前环境信息
    print(f"📍 Python: {sys.version.split()[0]}")
    print(f"📁 目录: {os.getcwd()}")
    
    # 使用pip freeze获取包列表
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True, check=True)
        packages = result.stdout.strip().split('\n')
        packages = [pkg for pkg in packages if pkg.strip()]
        
        print(f"📦 找到 {len(packages)} 个包")
        
        # 写入requirements.txt
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(f"# EasyTSAD项目依赖包\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Python版本: {sys.version.split()[0]}\n")
            f.write(f"# 包数量: {len(packages)}\n\n")
            
            for package in sorted(packages):
                f.write(f"{package}\n")
        
        print("✅ 成功生成 requirements.txt")
        
        # 显示前几个包
        print("\n📋 主要包预览:")
        for i, pkg in enumerate(sorted(packages)[:10]):
            print(f"   {pkg}")
        if len(packages) > 10:
            print(f"   ... 还有 {len(packages) - 10} 个包")
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")

def generate_core_requirements():
    """生成核心包的requirements.txt"""
    print("🎯 生成核心包requirements.txt...")
    
    # 定义核心包关键词
    core_keywords = [
        'torch', 'tensorflow', 'keras', 'numpy', 'pandas', 'scipy', 
        'scikit-learn', 'matplotlib', 'seaborn', 'tqdm', 'requests',
        'pyyaml', 'pillow', 'opencv', 'jupyter', 'notebook', 'ipython'
    ]
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True, check=True)
        all_packages = result.stdout.strip().split('\n')
        
        # 过滤核心包
        core_packages = []
        for pkg in all_packages:
            if pkg.strip():
                pkg_name = pkg.split('==')[0].lower()
                if any(keyword in pkg_name for keyword in core_keywords):
                    core_packages.append(pkg)
        
        print(f"📦 找到 {len(core_packages)} 个核心包 (总共 {len(all_packages)} 个)")
        
        # 写入requirements_core.txt
        with open("requirements_core.txt", "w", encoding="utf-8") as f:
            f.write(f"# EasyTSAD项目核心依赖包\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 核心包数量: {len(core_packages)}\n\n")
            
            for package in sorted(core_packages):
                f.write(f"{package}\n")
        
        print("✅ 成功生成 requirements_core.txt")
        print("\n📋 核心包列表:")
        for pkg in sorted(core_packages):
            print(f"   {pkg}")
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")

if __name__ == "__main__":
    print("🎯 ========== Requirements 快速生成工具 ==========")
    print("1. 生成完整的requirements.txt")
    print("2. 生成核心包requirements_core.txt")
    print("3. 同时生成两个文件")
    
    choice = input("\n请选择 (1-3, 默认3): ").strip() or "3"
    
    if choice == "1":
        generate_requirements_simple()
    elif choice == "2":
        generate_core_requirements()
    elif choice == "3":
        generate_requirements_simple()
        print("\n" + "-"*50)
        generate_core_requirements()
    else:
        print("❌ 无效选择")
    
    print("\n🎉 完成！") 