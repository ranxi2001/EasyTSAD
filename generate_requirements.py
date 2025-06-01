#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成requirements.txt文件的工具
支持多种方法获取包版本信息
"""

import subprocess
import sys
import os
import json
from pathlib import Path
import importlib.metadata
import pkg_resources
from typing import List, Dict, Set

class RequirementsGenerator:
    def __init__(self):
        self.output_file = "requirements.txt"
        self.conda_file = "environment.yml"
        
    def method_1_pip_freeze(self) -> List[str]:
        """方法1: 使用pip freeze命令"""
        print("🔍 方法1: 使用pip freeze获取包信息...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                                  capture_output=True, text=True, check=True)
            packages = result.stdout.strip().split('\n')
            packages = [pkg for pkg in packages if pkg.strip()]
            print(f"   ✅ 找到 {len(packages)} 个pip包")
            return packages
        except subprocess.CalledProcessError as e:
            print(f"   ❌ pip freeze失败: {e}")
            return []

    def method_2_conda_list(self) -> List[str]:
        """方法2: 使用conda list命令"""
        print("🔍 方法2: 使用conda list获取包信息...")
        try:
            result = subprocess.run(["conda", "list", "--export"], 
                                  capture_output=True, text=True, check=True)
            packages = []
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('#'):
                    # 转换conda格式到pip格式
                    parts = line.split('=')
                    if len(parts) >= 2:
                        pkg_name = parts[0]
                        version = parts[1]
                        packages.append(f"{pkg_name}=={version}")
            print(f"   ✅ 找到 {len(packages)} 个conda包")
            return packages
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"   ❌ conda list失败: {e}")
            return []

    def method_3_importlib(self) -> List[str]:
        """方法3: 使用importlib.metadata"""
        print("🔍 方法3: 使用importlib.metadata获取包信息...")
        try:
            packages = []
            for dist in importlib.metadata.distributions():
                name = dist.metadata['Name']
                version = dist.version
                packages.append(f"{name}=={version}")
            print(f"   ✅ 找到 {len(packages)} 个包")
            return packages
        except Exception as e:
            print(f"   ❌ importlib.metadata失败: {e}")
            return []

    def method_4_pkg_resources(self) -> List[str]:
        """方法4: 使用pkg_resources"""
        print("🔍 方法4: 使用pkg_resources获取包信息...")
        try:
            packages = []
            for dist in pkg_resources.working_set:
                packages.append(f"{dist.project_name}=={dist.version}")
            print(f"   ✅ 找到 {len(packages)} 个包")
            return packages
        except Exception as e:
            print(f"   ❌ pkg_resources失败: {e}")
            return []

    def get_core_packages(self) -> Set[str]:
        """获取项目核心依赖包"""
        core_packages = {
            # 深度学习框架
            'torch', 'pytorch', 'tensorflow', 'keras',
            # 数据科学
            'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn',
            # 工具库
            'tqdm', 'requests', 'pyyaml', 'pillow', 'opencv-python',
            # Jupyter
            'jupyter', 'notebook', 'ipython',
            # 其他常用
            'flask', 'django', 'fastapi', 'streamlit'
        }
        return core_packages

    def filter_packages(self, packages: List[str], include_all: bool = False) -> List[str]:
        """过滤包列表"""
        if include_all:
            return packages
            
        core_packages = self.get_core_packages()
        filtered = []
        
        for pkg in packages:
            pkg_name = pkg.split('==')[0].lower().replace('_', '-')
            # 检查是否为核心包
            if any(core in pkg_name for core in core_packages):
                filtered.append(pkg)
                
        return filtered

    def generate_conda_environment(self, packages: List[str]):
        """生成conda environment.yml文件"""
        print(f"\n📝 生成conda环境文件: {self.conda_file}")
        
        env_data = {
            'name': 'EasyTSAD',
            'channels': ['defaults', 'conda-forge', 'pytorch'],
            'dependencies': []
        }
        
        conda_packages = []
        pip_packages = []
        
        for pkg in packages:
            if '==' in pkg:
                name, version = pkg.split('==', 1)
                # 常见的conda包
                if name.lower() in ['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'pytorch', 'tensorflow']:
                    conda_packages.append(f"{name}={version}")
                else:
                    pip_packages.append(pkg)
            else:
                pip_packages.append(pkg)
        
        env_data['dependencies'].extend(conda_packages)
        if pip_packages:
            env_data['dependencies'].append({'pip': pip_packages})
        
        try:
            import yaml
            with open(self.conda_file, 'w', encoding='utf-8') as f:
                yaml.dump(env_data, f, default_flow_style=False, allow_unicode=True)
            print(f"   ✅ 成功生成: {self.conda_file}")
        except ImportError:
            print("   ⚠️  需要安装pyyaml包来生成yml文件")
        except Exception as e:
            print(f"   ❌ 生成yml文件失败: {e}")

    def generate_requirements(self, method: str = "auto", include_all: bool = False, 
                            generate_conda: bool = True):
        """生成requirements.txt文件"""
        print("🚀 开始生成requirements.txt文件...")
        print(f"📁 当前工作目录: {os.getcwd()}")
        
        all_packages = []
        
        if method == "auto":
            # 尝试所有方法，选择最好的结果
            methods = [
                self.method_1_pip_freeze,
                self.method_2_conda_list,
                self.method_3_importlib,
                self.method_4_pkg_resources
            ]
            
            for method_func in methods:
                packages = method_func()
                if packages and len(packages) > len(all_packages):
                    all_packages = packages
                    
        elif method == "pip":
            all_packages = self.method_1_pip_freeze()
        elif method == "conda":
            all_packages = self.method_2_conda_list()
        elif method == "importlib":
            all_packages = self.method_3_importlib()
        elif method == "pkg_resources":
            all_packages = self.method_4_pkg_resources()
        else:
            print(f"❌ 未知方法: {method}")
            return
        
        if not all_packages:
            print("❌ 未能获取到任何包信息")
            return
        
        # 去重和排序
        unique_packages = list(set(all_packages))
        unique_packages.sort(key=lambda x: x.lower())
        
        print(f"\n📊 包统计信息:")
        print(f"   总包数: {len(unique_packages)}")
        
        # 过滤包
        if not include_all:
            filtered_packages = self.filter_packages(unique_packages, include_all)
            print(f"   核心包数: {len(filtered_packages)}")
            final_packages = filtered_packages if filtered_packages else unique_packages
        else:
            final_packages = unique_packages
        
        # 生成requirements.txt
        print(f"\n📝 生成requirements文件: {self.output_file}")
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("# EasyTSAD项目依赖包\n")
                f.write(f"# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 包数量: {len(final_packages)}\n\n")
                
                for package in final_packages:
                    f.write(f"{package}\n")
            
            print(f"   ✅ 成功生成: {self.output_file}")
            print(f"   📦 包含 {len(final_packages)} 个包")
            
        except Exception as e:
            print(f"   ❌ 生成requirements.txt失败: {e}")
            return
        
        # 生成conda环境文件
        if generate_conda:
            self.generate_conda_environment(final_packages)
        
        # 显示前10个包
        print(f"\n📋 前10个包预览:")
        for i, pkg in enumerate(final_packages[:10]):
            print(f"   {i+1:2d}. {pkg}")
        if len(final_packages) > 10:
            print(f"   ... 还有 {len(final_packages) - 10} 个包")

    def show_current_env_info(self):
        """显示当前环境信息"""
        print("🔍 当前环境信息:")
        print(f"   Python版本: {sys.version}")
        print(f"   Python路径: {sys.executable}")
        print(f"   工作目录: {os.getcwd()}")
        
        # 检查是否在conda环境中
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            print(f"   Conda环境: {conda_env}")
        else:
            print("   Conda环境: 未检测到")

def main():
    generator = RequirementsGenerator()
    
    print("🎯 ========== Requirements.txt 自动生成工具 ==========")
    generator.show_current_env_info()
    
    # 用户选择
    print("\n📝 选择生成方式:")
    print("1. 自动选择最佳方法 (推荐)")
    print("2. 仅使用pip freeze")
    print("3. 仅使用conda list")
    print("4. 仅使用importlib.metadata")
    print("5. 仅使用pkg_resources")
    
    try:
        choice = input("\n请选择 (1-5, 默认1): ").strip() or "1"
        
        method_map = {
            "1": "auto",
            "2": "pip", 
            "3": "conda",
            "4": "importlib",
            "5": "pkg_resources"
        }
        
        method = method_map.get(choice, "auto")
        
        # 是否包含所有包
        include_all_input = input("是否包含所有包? (y/N, 默认只包含核心包): ").strip().lower()
        include_all = include_all_input in ['y', 'yes', '是']
        
        # 是否生成conda环境文件
        conda_input = input("是否生成conda environment.yml? (Y/n, 默认是): ").strip().lower()
        generate_conda = conda_input not in ['n', 'no', '否']
        
        print("\n" + "="*60)
        generator.generate_requirements(method=method, include_all=include_all, 
                                      generate_conda=generate_conda)
        print("="*60)
        print("🎉 完成！")
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main() 