import os
import sys
import argparse
import matplotlib.pyplot as plt

# 将 src 目录加入到 Python 模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import read_image_chinese_path, preprocess_image, save_image_chinese_path
from engine import LightingEstimator

def main(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: 找不到输入图像 {input_path}")
        return

    # 1. 预处理 (模块A)
    try:
        raw_img = read_image_chinese_path(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"成功读取图像: {input_path}, 原始尺寸: {raw_img.shape}")
    
    # 缩放并归一化至 [0, 1] RGB
    img_rgb = preprocess_image(raw_img, target_width=512)
    print("图像预处理完成。")

    # 2. 光照估计 (模块B & C)
    print("开始预测环境光照图...")
    estimator = LightingEstimator(pano_height=256, pano_width=512)
    predicted_pano = estimator.estimate(img_rgb)
    print("环境光照图预测完成。")

    # 3. 自动保存结果
    if output_path is None:
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        
        if not os.path.exists(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
        output_path = os.path.join(output_dir, f"{name}_envmap.png")
    
    try:
        save_image_chinese_path(output_path, predicted_pano)
        print(f"预测结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")

    # 4. 可视化对比 (模块D)
    plt.figure(figsize=(12, 6))
    
    # 显示原始预处理后的图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image (LDR)")
    plt.axis('off')
    
    # 显示预测的全景图
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_pano)
    plt.title("Predicted Environment Map (HDR-like)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从单幅 LDR 图像预测全景环境光照图")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出全景图路径 (可选)")
    
    args = parser.parse_args()
    
    main(args.input, args.output)
