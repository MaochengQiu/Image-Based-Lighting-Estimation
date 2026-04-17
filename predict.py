import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import rcParams


# ==================== 📊 表格生成函数 ====================
def save_metrics_table_png(light_dir, intensity, dynamic_range, contrast, softness, shadow_coherence):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # 设置字体，优先使用微软雅黑，降级为黑体
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    rcParams['axes.unicode_minus'] = False

    # 优化数据格式：增加中英文对照，优化小数位
    data = [
        ["主光源方向 (Direction)", f"[{light_dir[0]:.2f}, {light_dir[1]:.2f}, {light_dir[2]:.2f}]", "3D空间方向向量"],
        ["光照强度 (Intensity)", f"{intensity:.3f}", "全局亮度特征 (0-1)"],
        ["动态范围 (Dynamic Range)", f"{dynamic_range:.3f}", "极值亮度差异"],
        ["全局对比度 (Contrast)", f"{contrast:.3f}", "光照层次与深度"],
        ["光照柔和度 (Softness)", f"{softness:.2e}", "环境图方差 (越小越柔)"],  # 使用科学计数法
        ["阴影一致性 (Coherence)", f"{shadow_coherence:.3f}", "梯度场平均模长"]
    ]

    # 调整画布比例，适合 PPT 宽屏
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')

    # 创建表格，使用 bbox 占满画布空间
    table = ax.table(
        cellText=data,
        colLabels=["评估指标 (Metrics)", "测量数值 (Values)", "物理意义 (Description)"],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)

    # =============== 核心：绘制学术三线表 ===============
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('none')  # 先移除所有默认边框

        # 表头 (第一条和第二条线)
        if row == 0:
            cell.set_text_props(weight='bold', color='#2c3e50')
            cell.visible_edges = 'BT'  # 顶部(Top)和底部(Bottom)画线
            cell.set_edgecolor('#000000')
            cell.set_linewidth(1.5)
            cell.set_facecolor('#f8f9fa')  # 表头浅灰色背景

        # 表底 (第三条线)
        elif row == len(data):
            cell.visible_edges = 'B'  # 底部(Bottom)画线
            cell.set_edgecolor('#000000')
            cell.set_linewidth(1.5)

        # 数据行：添加斑马纹以增强 PPT 上的可读性
        else:
            if row % 2 == 0:
                cell.set_facecolor('#fdfdfd')
            else:
                cell.set_facecolor('#f4f6f8')

    plt.title("图 X：环境光照估计量化评估结果", fontsize=15, weight='bold', pad=25, color='#1a252f')

    output_path = "lighting_metrics_table_academic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f"✔ 学术级三线表已生成: {output_path}")
# ==================== 🌍 主算法 ====================
def estimate_envmap_with_cnn(image_path):

    # ==================== 读取图像 ====================
    raw = np.fromfile(r"C:\Users\lenovo\Documents\INNOVATION\lighting\exp1.jpg", dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        print("图片读取失败")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    I = torch.from_numpy(gray)[None, None, :, :]

    # ==================== CNN增强 ====================
    lowpass_kernel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=torch.float32)
    lowpass_kernel /= lowpass_kernel.sum()
    lowpass = F.conv2d(I, lowpass_kernel[None, None], padding=1)

    highpass_kernel = torch.tensor([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=torch.float32)
    highpass = F.conv2d(I, highpass_kernel[None, None], padding=1)

    feat = (lowpass * 0.85 + highpass * 0.25).clamp(0, 1)

    # ==================== 大尺度光照 ====================
    kernel_size = 51
    gaussian = cv2.getGaussianKernel(kernel_size, kernel_size / 3)
    gaussian = torch.from_numpy(gaussian @ gaussian.T).float()[None, None]
    env_low = F.conv2d(feat, gaussian, padding=kernel_size // 2)

    # ==================== 环境图生成 ====================
    env_h, env_w = 256, 512
    env = np.zeros((env_h, env_w), dtype=np.float32)

    for y in range(env_h):
        for x in range(env_w):
            img_x = int(x / env_w * w)
            img_y = int(y / env_h * h)
            env[y, x] = env_low[0, 0, img_y, img_x].item()

    # ==================== tone mapping ====================
    env = env / (1 + env)
    env = np.power(env, 0.7)
    env_rgb = np.dstack([env, env, env])

    # ==================== 📊 量化分析 ====================
    max_idx = np.unravel_index(np.argmax(env), env.shape)
    max_y, max_x = max_idx

    theta = (max_x / env_w) * 2 * np.pi
    phi = (max_y / env_h) * np.pi

    light_dir = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

    intensity = float(env.max())
    dynamic_range = float(env.max() - env.min())
    contrast = dynamic_range / (env.max() + 1e-6)
    softness = float(np.var(env))

    grad_x = cv2.Sobel(env, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(env, cv2.CV_32F, 0, 1, ksize=5)
    shadow_coherence = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

    # ==================== 📄 表格输出 ====================
    save_metrics_table_png(light_dir, intensity, dynamic_range, contrast, softness, shadow_coherence)

    # ==================== 🖼 可视化 ====================
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Input (LDR)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(env_rgb)
    plt.title("CNN-enhanced Environment Map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("cnn_envmap.png", dpi=300)
    print("✔ 环境图已生成: cnn_envmap.png")

    plt.show()


# ==================== 🚀 运行 ====================
estimate_envmap_with_cnn(r"C:\Users\lenovo\Documents\INNOVATION\lighting\exp1.jpg")