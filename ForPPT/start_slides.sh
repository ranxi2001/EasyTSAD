#!/bin/bash

# =================================================================
# LightMMoE Slides 启动脚本
# 用于自动启动LightMMoE算法演示文稿
# =================================================================

echo "🎯 ========== LightMMoE Slides 启动脚本 =========="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Node.js是否安装
echo -e "${BLUE}📋 正在检查环境...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ 错误: 未检测到Node.js${NC}"
    echo -e "${YELLOW}请先安装Node.js: https://nodejs.org/${NC}"
    exit 1
fi

# 检查npm是否安装
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ 错误: 未检测到npm${NC}"
    echo -e "${YELLOW}请先安装npm${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Node.js 和 npm 环境正常${NC}"

# 检查slidev是否已安装
echo -e "${BLUE}📦 检查Slidev安装状态...${NC}"
if ! command -v slidev &> /dev/null; then
    echo -e "${YELLOW}⚠️  Slidev未安装，正在自动安装...${NC}"
    echo -e "${BLUE}🔧 执行: npm install -g @slidev/cli${NC}"
    
    if npm install -g @slidev/cli; then
        echo -e "${GREEN}✅ Slidev安装成功！${NC}"
    else
        echo -e "${RED}❌ Slidev安装失败${NC}"
        echo -e "${YELLOW}💡 请尝试手动安装: npm install -g @slidev/cli${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Slidev已安装${NC}"
fi

# 检查slides文件是否存在
SLIDES_FILE="LightMMoE_slides.md"
if [ ! -f "$SLIDES_FILE" ]; then
    echo -e "${RED}❌ 错误: 找不到演示文件 $SLIDES_FILE${NC}"
    echo -e "${YELLOW}请确保该脚本在正确的目录中运行${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 演示文件检查通过${NC}"
echo ""

# 显示演示信息
echo -e "${BLUE}🎯 ========== LightMMoE算法演示信息 ==========${NC}"
echo -e "${GREEN}📄 演示文件:${NC} $SLIDES_FILE"
echo -e "${GREEN}📊 演示内容:${NC} 轻量级多专家混合异常检测算法"
echo -e "${GREEN}🎭 主要特色:${NC}"
echo "   • 18页完整技术演示"
echo "   • 基于真实实验数据 (Point F1: 93.4%)"
echo "   • 代码语法高亮展示"
echo "   • Mermaid架构图"
echo "   • 响应式布局设计"
echo "   • 流畅动画过渡效果"
echo ""

# 启动确认
echo -e "${YELLOW}🚀 准备启动演示...${NC}"
echo -e "${BLUE}💡 启动后请在浏览器访问: http://localhost:3030${NC}"
echo ""

read -p "按Enter键启动演示，或Ctrl+C取消: "

# 启动slidev
echo -e "${GREEN}🎬 正在启动LightMMoE Slides演示...${NC}"
echo -e "${BLUE}📱 浏览器将自动打开，如未打开请手动访问: http://localhost:3030${NC}"
echo ""

# 启动slidev并自动打开浏览器
if slidev "$SLIDES_FILE" --open; then
    echo -e "${GREEN}✅ 演示启动成功！${NC}"
else
    echo -e "${RED}❌ 演示启动失败${NC}"
    echo -e "${YELLOW}💡 请检查slides文件格式或尝试手动启动: slidev $SLIDES_FILE${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}🎯 ========== 使用说明 ==========${NC}"
echo -e "${GREEN}⌨️  快捷键:${NC}"
echo "   • 左右箭头键: 上一页/下一页"
echo "   • 空格键: 下一页"
echo "   • F: 全屏模式"
echo "   • O: 概览模式"
echo "   • D: 深色模式切换"
echo ""
echo -e "${GREEN}🌐 演示控制:${NC}"
echo "   • 演讲者模式: 按S键"
echo "   • 录制模式: 按R键"
echo "   • 绘图模式: 按D键"
echo ""
echo -e "${YELLOW}📝 注意: 要停止演示，请按 Ctrl+C${NC}"
echo ""
echo -e "${GREEN}🎉 享受您的LightMMoE算法演示！${NC}" 