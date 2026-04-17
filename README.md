# Lighting Estimation Project

本项目为中期答辩所需的“基线系统”。核心逻辑是：输入一张室内 LDR 照片，通过图像处理与启发式算法，定位光源并生成一张 360° 的全景 HDR-like 环境图（Latitude-Longitude Map）。

## 技术栈
- Language: Python 3.9+
- Core Libs: PyTorch, OpenCV, NumPy, Matplotlib

## 目录结构
- `/inputs`: 存放测试照片（输入 LDR 图像）
- `/outputs`: 存放生成的预测环境图
- `/src`: 包含核心算法模块
  - `utils.py`: 图像读取（支持中文路径）、预处理及图像保存
  - `engine.py`: 核心光照提取逻辑、环境图生成及物理衰减模糊处理
- `main.py`: 程序入口主循环，支持可视化和自动保存

## 运行方式
确保安装所需的依赖项：
```bash
pip install torch numpy opencv-python matplotlib
```

运行 `main.py` 脚本，传入一张输入图像路径即可：
```bash
python main.py -i inputs/test_image.jpg
```

运行后会在屏幕上弹出“原始图像”与“预测环境图”的对比窗口，同时结果会自动保存在 `outputs/` 目录下。
