@echo off
chcp 65001 >nul
cls

echo 🎯 ========== LightMMoE Slides 启动脚本 ==========
echo.

:: 检查Node.js是否安装
echo 📋 正在检查环境...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未检测到Node.js
    echo 💡 请先安装Node.js: https://nodejs.org/
    pause
    exit /b 1
)

:: 检查npm是否安装
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未检测到npm
    echo 💡 请先安装npm
    pause
    exit /b 1
)

echo ✅ Node.js 和 npm 环境正常

:: 检查slidev是否已安装
echo 📦 检查Slidev安装状态...
slidev --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Slidev未安装，正在自动安装...
    echo 🔧 执行: npm install -g @slidev/cli
    
    npm install -g @slidev/cli
    if %errorlevel% neq 0 (
        echo ❌ Slidev安装失败
        echo 💡 请尝试手动安装: npm install -g @slidev/cli
        pause
        exit /b 1
    )
    echo ✅ Slidev安装成功！
) else (
    echo ✅ Slidev已安装
)

:: 检查slides文件是否存在
set SLIDES_FILE=LightMMoE_slides.md
if not exist "%SLIDES_FILE%" (
    echo ❌ 错误: 找不到演示文件 %SLIDES_FILE%
    echo 💡 请确保该脚本在正确的目录中运行
    pause
    exit /b 1
)

echo ✅ 演示文件检查通过
echo.

:: 显示演示信息
echo 🎯 ========== LightMMoE算法演示信息 ==========
echo 📄 演示文件: %SLIDES_FILE%
echo 📊 演示内容: 轻量级多专家混合异常检测算法
echo 🎭 主要特色:
echo    • 18页完整技术演示
echo    • 基于真实实验数据 (Point F1: 93.4%%)
echo    • 代码语法高亮展示
echo    • Mermaid架构图
echo    • 响应式布局设计
echo    • 流畅动画过渡效果
echo.

:: 启动确认
echo 🚀 准备启动演示...
echo 💡 启动后请在浏览器访问: http://localhost:3030
echo.
pause

:: 启动slidev
echo 🎬 正在启动LightMMoE Slides演示...
echo 📱 浏览器将自动打开，如未打开请手动访问: http://localhost:3030
echo.

:: 启动slidev并自动打开浏览器
slidev "%SLIDES_FILE%" --open
if %errorlevel% neq 0 (
    echo ❌ 演示启动失败
    echo 💡 请检查slides文件格式或尝试手动启动: slidev %SLIDES_FILE%
    pause
    exit /b 1
)

echo.
echo 🎯 ========== 使用说明 ==========
echo ⌨️  快捷键:
echo    • 左右箭头键: 上一页/下一页
echo    • 空格键: 下一页
echo    • F: 全屏模式
echo    • O: 概览模式
echo    • D: 深色模式切换
echo.
echo 🌐 演示控制:
echo    • 演讲者模式: 按S键
echo    • 录制模式: 按R键
echo    • 绘图模式: 按D键
echo.
echo 📝 注意: 要停止演示，请按 Ctrl+C
echo.
echo 🎉 享受您的LightMMoE算法演示！
echo.
echo 按任意键关闭此窗口...
pause >nul 