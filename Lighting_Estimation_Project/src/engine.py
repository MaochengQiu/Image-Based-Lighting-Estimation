import numpy as np
import cv2
import torch
import math

class LightingEstimator:
    def __init__(self, pano_height=256, pano_width=512):
        self.pano_height = pano_height
        self.pano_width = pano_width

    def estimate(self, img_rgb):
        """
        核心光照估计逻辑
        :param img_rgb: 输入的归一化 RGB 图像 (H, W, 3), [0, 1]
        :return: 预测的全景环境光照图 (pano_height, pano_width, 3), [0, 1]
        """
        # 1. 提取环境光 (Ambient Light)
        ambient_color = self.extract_ambient_light(img_rgb)
        
        # 2. 提取高光光源 (Highlight Sources)
        light_sources = self.extract_highlight_sources(img_rgb)
        
        # 3. 生成基础全景图并映射光源
        pano_map = self.generate_env_map(ambient_color, light_sources, img_rgb.shape[:2])
        
        # 4. 渲染与平滑处理 (物理衰减模糊与球面畸变模拟)
        final_pano = self.render_and_smooth(pano_map)
        
        return final_pano

    def extract_ambient_light(self, img_rgb):
        """
        提取图像的全局平均颜色作为 Ambient Light
        """
        # 简单使用全局平均
        ambient_color = np.mean(img_rgb, axis=(0, 1))
        # 降低环境光强度，让高光更突出
        return ambient_color * 0.5

    def extract_highlight_sources(self, img_rgb):
        """
        提取画面中的高亮光源区域
        计算光斑的质心及面积
        :return: list of dicts [{'centroid': (x, y), 'area': float, 'color': (r, g, b)}]
        """
        # 转换为灰度图进行亮度分析
        gray_img = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 使用高百分比截断法提取高亮区域 (取最亮的 5%)
        threshold_val = np.percentile(gray_img, 95)
        _, mask = cv2.threshold(gray_img, threshold_val, 255, cv2.THRESH_BINARY)
        
        # 连通域分析提取独立光源
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        light_sources = []
        for i in range(1, num_labels):  # 0 是背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 10:  # 过滤太小的噪点
                continue
                
            cx, cy = centroids[i]
            
            # 获取该区域的平均颜色
            mask_i = (labels == i)
            color = np.mean(img_rgb[mask_i], axis=0)
            
            # 提高高光亮度，模拟 HDR 效果
            color = np.clip(color * 2.0, 0, 1.0)
            
            light_sources.append({
                'centroid': (cx, cy),
                'area': area,
                'color': color
            })
            
        return light_sources

    def generate_env_map(self, ambient_color, light_sources, img_shape):
        """
        将提取的光源映射到球面全景图中
        """
        # 初始化为环境光颜色
        pano_map = np.ones((self.pano_height, self.pano_width, 3), dtype=np.float32) * ambient_color
        
        img_h, img_w = img_shape
        
        # 简单启发式映射：将 2D 图像中心的相对位置映射到全景图的正前方 (经度范围: -pi/2 到 pi/2, 纬度范围: -pi/4 到 pi/4)
        for source in light_sources:
            cx, cy = source['centroid']
            area = source['area']
            color = source['color']
            
            # 将 2D 坐标归一化到 [-1, 1]
            nx = (cx / img_w) * 2 - 1
            ny = (cy / img_h) * 2 - 1
            
            # 映射到经纬度 (简单假设前方视野范围)
            theta = nx * (np.pi / 2)  # 经度 [-pi/2, pi/2]
            phi = ny * (np.pi / 4)    # 纬度 [-pi/4, pi/4]
            
            # 经纬度转全景图像素坐标
            pano_x = int((theta / (2 * np.pi) + 0.5) * self.pano_width)
            pano_y = int((phi / np.pi + 0.5) * self.pano_height)
            
            # 确保在边界内
            pano_x = max(0, min(self.pano_width - 1, pano_x))
            pano_y = max(0, min(self.pano_height - 1, pano_y))
            
            # 根据面积估算光源在全景图中的大小 (简单比例)
            radius = int(math.sqrt(area) * (self.pano_width / img_w) * 0.5)
            radius = max(2, min(20, radius))
            
            # 绘制光源 (在基础图上叠加)
            cv2.circle(pano_map, (pano_x, pano_y), radius, color.tolist(), -1)
            
        return pano_map

    def render_and_smooth(self, pano_map):
        """
        对生成的全景图应用物理启发的衰减模糊
        模拟球面畸变 (靠近两极的光斑呈现经纬度拉伸效果)
        """
        # 为了实现两极拉伸的模糊效果，我们可以在纵向上采用不同的高斯模糊核，或者直接使用各向异性模糊
        # 这里使用一种简单有效的近似方法：根据纬度调整水平向模糊核的大小
        
        result = np.copy(pano_map)
        
        # 基础全局模糊 (物理衰减)
        result = cv2.GaussianBlur(result, (15, 15), 5.0)
        
        # 模拟球面畸变：两极拉伸
        # 将全景图分块，按行进行不同程度的水平模糊
        rows = self.pano_height
        for y in range(rows):
            # 计算当前纬度对应的高斯核水平尺寸
            # 赤道 (y = rows/2) 处模糊最小，两极 (y=0, y=rows-1) 处模糊最大
            lat_norm = abs(y - rows / 2) / (rows / 2) # [0, 1]
            
            # 拉伸因子，两极拉伸更严重
            stretch_factor = 1.0 / max(0.01, math.cos(lat_norm * (math.pi / 2) * 0.9))
            
            ksize_x = int(5 * stretch_factor)
            if ksize_x % 2 == 0:
                ksize_x += 1
            
            # 限制最大核大小
            ksize_x = min(101, ksize_x)
            
            if ksize_x > 3:
                # 对当前行进行一维水平高斯模糊
                row_img = result[y:y+1, :, :]
                blurred_row = cv2.GaussianBlur(row_img, (ksize_x, 1), sigmaX=ksize_x/3.0)
                result[y:y+1, :, :] = blurred_row
                
        # 再次全局轻微模糊平滑过渡
        final_pano = cv2.GaussianBlur(result, (7, 7), 2.0)
        
        # 确保像素值不越界
        final_pano = np.clip(final_pano, 0, 1.0)
        
        return final_pano
