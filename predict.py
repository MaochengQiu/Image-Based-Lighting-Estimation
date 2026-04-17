import numpy as np
import cv2
import matplotlib.pyplot as plt


def advanced_lighting_estimation(image_path):
    # 1. 兼容中文路径读取
    try:
        raw_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    except Exception as e:
        return print(f"读取文件出错: {e}")

    if img is None:
        return print(f"错误：无法读取图片\n{image_path}")

    # 将 BGR 转换为 RGB 以便后续处理和显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化环境图 (256x512)
    env_map = np.zeros((256, 512, 3), dtype=np.float32)

    # ================= 核心优化 1：提取全局环境光 (Ambient Light) =================
    # 计算全图平均颜色，并将其作为环境基础色调（稍微调暗）
    ambient_color = cv2.mean(img_rgb)[:3]
    ambient_color = np.array(ambient_color) / 255.0 * 0.15
    env_map[:, :, :] = ambient_color

    # ================= 核心优化 2：多光源与面积光提取 =================
    # 使用阈值法提取图中所有高亮区域（模拟多个光源或窗户）
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历找到的前 3 个最大亮斑（防止反光干扰）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    for cnt in contours:
        # 计算光斑中心点
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # ================= 核心优化 3：光源色彩提取 =================
            # 获取光源中心及其周围的平均颜色（暖光/冷光）
            light_color = img_rgb[cy, cx] / 255.0

            # 映射到球面坐标（简化的墨卡托投影映射）
            map_x = int((cx / img.shape[1]) * 512)
            map_y = int((cy / img.shape[0]) * 256)

            # 根据光斑在原图的面积，决定在环境图上的光晕大小
            radius = int(np.sqrt(cv2.contourArea(cnt)) / 2) + 10

            # 绘制光源，并做区域高斯模糊模拟光线衰减
            cv2.circle(env_map, (map_x, map_y), radius, light_color.tolist(), -1)

    # ================= 核心优化 4：HDR 渲染模拟 =================
    # 全局高斯模糊，模拟真实 Environment Map 的柔和过渡
    env_map = cv2.GaussianBlur(env_map, (61, 61), 0)
    # 模拟 HDR 曝光过度效果 (Clip 限制在 0-1)
    env_map = np.clip(env_map * 1.5, 0, 1)

    # ================= 结果可视化 =================
    plt.figure(figsize=(14, 6))

    # 显示原图
    plt.subplot(1, 2, 1)
    plt.title("Input Scene (LDR)", fontsize=14)
    plt.imshow(img_rgb)
    plt.axis('off')

    # 显示高精度环境图
    plt.subplot(1, 2, 2)
    plt.title("Advanced Environment Map (Baseline)", fontsize=14)
    plt.imshow(env_map)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("advanced_result.png", dpi=300)  # 保存高清图供 PPT 使用
    print("高级优化版运行成功！结果已保存为 advanced_result.png")
    plt.show()


# 替换为你实际的图片路径
target_path = r"C:\Users\lenovo\Documents\INNOVATION\lighting\exp1.jpg"
advanced_lighting_estimation(target_path)