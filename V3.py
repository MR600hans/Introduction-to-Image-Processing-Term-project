import cv2
import numpy as np
import os

def compute_foreground_mask(edges, min_area=4000):
    """
    使用形態學操作生成前景物體的掩膜，支援多個物體。

    :param edges: 邊緣圖（0和255）
    :param min_area: 最小面積閾值，過濾掉較小的雜訊區域
    :return: 前景掩膜，值為0或1
    """
    # 將邊緣圖轉換為二值圖
    binary_edges = (edges / 255).astype(np.uint8)

    # 使用形態學操作連接和填充物體區域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(binary_edges, kernel, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 尋找所有連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # 建立前景掩膜
    foreground_mask = np.zeros_like(labels, dtype=np.uint8)

    # 遍歷所有連通區域，跳過背景（標籤0）
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = width / height

        # 根據面積和長寬比進行過濾
        if area >= min_area and 0.5 <= aspect_ratio <= 2.0:
            foreground_mask[labels == i] = 1

    return foreground_mask

def compute_focus_map_from_mask(foreground_mask):
    """
    根據前景掩膜生成焦點權重圖。

    :param foreground_mask: 前景掩膜，值為0或1
    :return: 焦點權重圖，值範圍在 [0,1]
    """
    # 計算距離轉換
    dist_transform = cv2.distanceTransform(1 - foreground_mask, cv2.DIST_L2, 5)

    # 正規化並取反
    focus_map = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    focus_map = 1 - focus_map

    # 平滑焦點權重圖
    focus_map = cv2.GaussianBlur(focus_map, (21, 21), sigmaX=15, sigmaY=15)

    return focus_map

def auto_adjust_blur_parameters(focus_map, base_max_blur=100, base_steps=5):
    """
    根据焦点权重图自动调整模糊参数。

    :param focus_map: 焦点权重图，值范围在 [0,1]
    :param base_max_blur: 基础最大模糊程度
    :param base_steps: 基础模糊累积的次数
    :return: 调整后的 max_blur 和 steps
    """
    # 计算焦外区域的比例（权重值低于阈值的区域）
    threshold = 0.5  # 可以根据需要调整
    out_of_focus_ratio = np.mean(focus_map < threshold)

    # 为了增加模糊效果，可以调整计算方式
    if out_of_focus_ratio < 0.1:
        # 焦外区域很小，需要更大的模糊程度
        adjusted_max_blur = int(base_max_blur * 3)
        adjusted_steps = int(base_steps * 2)
    else:
        adjusted_max_blur = int(base_max_blur / (out_of_focus_ratio + 0.01))  # 防止除以零
        adjusted_steps = int(base_steps / (out_of_focus_ratio + 0.01))

    # 设置模糊参数的上下限，防止过小或过大
    adjusted_max_blur = np.clip(adjusted_max_blur, 50, 300)
    adjusted_steps = np.clip(adjusted_steps, 5, 10)

    return adjusted_max_blur, adjusted_steps

def apply_strong_blur(image, focus_map, max_blur=100, steps=5):
    """
    对焦外区域应用更强的模糊，通过多次模糊累积效果。

    :param image: 原始影像
    :param focus_map: 焦点权重图，值范围在 [0,1]
    :param max_blur: 最大模糊程度
    :param steps: 模糊累积的次数
    :return: 处理后的影像
    """
    # 将焦点权重图扩展为3通道
    focus_map_3ch = cv2.merge([focus_map, focus_map, focus_map])

    # 确保数据类型为 float32
    image = image.astype(np.float32) / 255.0
    focus_map_3ch = focus_map_3ch.astype(np.float32)

    # 初始化结果影像
    result = image.copy()

    # 分阶段累积模糊效果
    for i in range(steps):
        # 计算当前阶段的模糊程度
        blur_amount = int(max_blur * (i + 1) / steps)
        if blur_amount % 2 == 0:
            blur_amount += 1  # 确保为奇数

        # 限制内核大小不超过图像尺寸
        ksize = min(blur_amount, image.shape[1] // 2 - 1, image.shape[0] // 2 - 1)
        if ksize % 2 == 0:
            ksize -= 1  # 确保为奇数

        if ksize < 3:
            ksize = 3  # 最小为3

        # 应用高斯模糊
        blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)

        # 计算当前阶段的权重图
        weight = ((1 - focus_map_3ch) ** 2) * ((i + 1) / steps)

        # 累积模糊效果
        result = result * (1 - weight) + blurred_image * weight

    # 确保结果在 [0,1] 范围内
    result = np.clip(result, 0, 1)

    # 转换回 uint8 类型
    result = (result * 255).astype(np.uint8)

    return result


def auto_canny(image, sigma=0.5):
    """
    根據圖像自動計算 Canny 邊緣檢測的閾值。

    :param image: 灰度圖像
    :param sigma: 控制閾值範圍的參數，默認為 0.33
    :return: 邊緣圖（0和255）
    """
    # 計算像素強度的中位數
    v = np.median(image)

    # 計算上下閾值
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # 應用 Canny 邊緣檢測
    edges = cv2.Canny(image, lower, upper)

    return edges

def main():
    # 载入输入影像
    input_image_path = 'DSC01197.jpg'  # 请替换为您的影像路径
    image = cv2.imread(input_image_path)

    if image is None:
        print("错误：无法载入影像。")
        return

    # 创建输出资料夹
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将影像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, 'gray_image.jpg'), gray_image)

    # 应用高斯模糊降低杂讯
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_folder, 'blurred_image.jpg'), blurred_image)

    # 使用自动 Canny 边缘检测
    edges = auto_canny(blurred_image)
    cv2.imwrite(os.path.join(output_folder, 'edges.jpg'), edges)

    # 生成前景掩膜
    foreground_mask = compute_foreground_mask(edges, min_area=4000)
    cv2.imwrite(os.path.join(output_folder, 'foreground_mask.jpg'), foreground_mask * 255)

    # 生成焦点权重图
    focus_map = compute_focus_map_from_mask(foreground_mask)
    cv2.imwrite(os.path.join(output_folder, 'focus_map.jpg'), (focus_map * 255).astype(np.uint8))

    # 自动调整模糊参数
    max_blur, steps = auto_adjust_blur_parameters(focus_map, base_max_blur=300, base_steps=10)
    print(f"自动调整后的模糊参数：max_blur={max_blur}, steps={steps}")

    # 应用强模糊到焦外区域
    result_image = apply_strong_blur(image, focus_map, max_blur=max_blur, steps=steps)

    # 保存最终结果
    cv2.imwrite(os.path.join(output_folder, 'result_image.jpg'), result_image)

    # 显示结果（可选）
    cv2.imshow('原始影像', image)
    cv2.imshow('焦点权重图', focus_map)
    cv2.imshow('处理后的影像', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()