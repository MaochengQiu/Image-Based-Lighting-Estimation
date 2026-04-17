<<<<<<< HEAD
import cv2
import numpy as np
import os

def read_image_chinese_path(file_path):
    """
    读取包含中文路径的图像
    :param file_path: 图像路径
    :return: BGR格式的图像 (NumPy array)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 使用 numpy 从文件读取字节流，再用 cv2 解码
    img_data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"无法读取图像，可能是格式不受支持: {file_path}")
        
    return img

def preprocess_image(img, target_width=512):
    """
    预处理图像：缩放至指定宽度并归一化
    :param img: 原始BGR图像
    :param target_width: 目标宽度 (默认512)
    :return: 处理后的 RGB 图像, 形状为 (H, W, 3), 范围 [0, 1]
    """
    h, w = img.shape[:2]
    
    # 计算缩放比例
    scale = target_width / w
    target_height = int(h * scale)
    
    # 缩放图像
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # BGR 转 RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # 归一化至 [0, 1]
    normalized_img = rgb_img.astype(np.float32) / 255.0
    
    return normalized_img

def save_image_chinese_path(file_path, img_rgb):
    """
    保存图像到包含中文的路径
    :param file_path: 保存路径
    :param img_rgb: RGB格式的图像 (范围 [0, 1] 或 [0, 255])
    """
    if img_rgb.max() <= 1.0:
        img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
    is_success, im_buf_arr = cv2.imencode('.png', img_bgr)
    if is_success:
        im_buf_arr.tofile(file_path)
    else:
        raise ValueError(f"保存图像失败: {file_path}")
=======
import cv2
import numpy as np
import os

def read_image_chinese_path(file_path):
    """
    读取包含中文路径的图像
    :param file_path: 图像路径
    :return: BGR格式的图像 (NumPy array)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 使用 numpy 从文件读取字节流，再用 cv2 解码
    img_data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"无法读取图像，可能是格式不受支持: {file_path}")
        
    return img

def preprocess_image(img, target_width=512):
    """
    预处理图像：缩放至指定宽度并归一化
    :param img: 原始BGR图像
    :param target_width: 目标宽度 (默认512)
    :return: 处理后的 RGB 图像, 形状为 (H, W, 3), 范围 [0, 1]
    """
    h, w = img.shape[:2]
    
    # 计算缩放比例
    scale = target_width / w
    target_height = int(h * scale)
    
    # 缩放图像
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # BGR 转 RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # 归一化至 [0, 1]
    normalized_img = rgb_img.astype(np.float32) / 255.0
    
    return normalized_img

def save_image_chinese_path(file_path, img_rgb):
    """
    保存图像到包含中文的路径
    :param file_path: 保存路径
    :param img_rgb: RGB格式的图像 (范围 [0, 1] 或 [0, 255])
    """
    if img_rgb.max() <= 1.0:
        img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
    is_success, im_buf_arr = cv2.imencode('.png', img_bgr)
    if is_success:
        im_buf_arr.tofile(file_path)
    else:
        raise ValueError(f"保存图像失败: {file_path}")
>>>>>>> adb55a4d09de0bdb71343240891ffd6e4faf4c93
