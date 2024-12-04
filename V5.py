import cv2
import numpy as np
import os

def compute_foreground_mask(edges, top_ratio=0.1):
    # 將邊緣圖轉換為二值圖
    binary_edges = (edges / 255).astype(np.uint8)

    # 使用形態學操作連接和填充物體區域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(binary_edges, kernel, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 尋找所有連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # 如果沒有連通區域，直接返回全黑的 foreground_mask
    if num_labels <= 1:
        print("未檢測到任何連通區域。")
        foreground_mask = np.zeros_like(labels, dtype=np.uint8)
        return foreground_mask

    # 計算所有連通區域的面積列表，排除背景（標籤0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    labels_indices = np.arange(1, num_labels)

    # 按面積降序排序
    sorted_indices = labels_indices[np.argsort(-areas)]
    sorted_areas = areas[np.argsort(-areas)]

    # 選擇面積排名前 N% 的連通區域
    num_top_regions = max(1, int(len(sorted_areas) * top_ratio))
    top_indices = sorted_indices[:num_top_regions]

    # 創建前景掩膜
    foreground_mask = np.zeros_like(labels, dtype=np.uint8)

    for i in top_indices:
        foreground_mask[labels == i] = 1

    return foreground_mask

def compute_focus_map_from_mask(foreground_mask):
    """
    根據前景掩膜生成焦點權重圖。

    :param foreground_mask: 前景掩膜，值為0或1
    :return: 焦點權重圖，值範圍在 [0,1]
    """
    # 計算距離變換
    dist_transform = cv2.distanceTransform(1 - foreground_mask, cv2.DIST_L2, 5)

    # 歸一化並取反
    focus_map = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    focus_map = 1 - focus_map

    # 平滑焦點權重圖
    focus_map = cv2.GaussianBlur(focus_map, (21, 21), sigmaX=15, sigmaY=15)

    return focus_map

def estimate_blur_amount(image, focus_map):
    """
    估計原始圖像的焦外模糊程度。

    :param image: 原始圖像
    :param focus_map: 焦點權重圖，值範圍在 [0,1]
    :return: 焦外區域的平均模糊程度
    """
    # 將圖像轉換為灰度圖
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 計算拉普拉斯變換
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # 計算焦外區域的掩膜（焦點權重較低的區域）
    out_of_focus_mask = (focus_map < 0.5).astype(np.uint8)

    # 計算焦外區域的方差（模糊程度）
    variance = cv2.mean(laplacian ** 2, mask=out_of_focus_mask)[0]

    # 返回模糊程度（方差的倒數）
    blur_amount = 1 / (variance + 1e-6)  # 防止除以零

    return blur_amount

def calculate_additional_blur(current_blur, desired_blur):
    """
    計算需要增加的模糊量。

    :param current_blur: 當前焦外區域的模糊程度
    :param desired_blur: 期望的焦外模糊程度
    :return: 需要增加的模糊量比例
    """
    # 計算增加的模糊量比例
    if current_blur < desired_blur:
        additional_blur_ratio = desired_blur / current_blur
    else:
        additional_blur_ratio = 1  # 已經達到或超過期望的模糊程度

    return additional_blur_ratio

def apply_strong_blur(image, focus_map, max_blur, steps):
    """
    對焦外區域應用更強的模糊，透過多次模糊累積效果。

    :param image: 原始影像
    :param focus_map: 焦點權重圖，值範圍在 [0,1]
    :param max_blur: 最大模糊程度
    :param steps: 模糊累積的次數
    :return: 處理後的影像
    """
    # 將焦點權重圖擴展為3通道
    focus_map_3ch = cv2.merge([focus_map, focus_map, focus_map])

    # 確保資料類型為 float32
    image = image.astype(np.float32) / 255.0
    focus_map_3ch = focus_map_3ch.astype(np.float32)

    # 初始化結果影像
    result = image.copy()

    # 分階段累積模糊效果
    for i in range(steps):
        print(f"應用模糊階段：{i + 1}/{steps}")
        # 計算當前階段的模糊程度
        blur_amount = int(max_blur * (i + 1) / steps)
        if blur_amount % 2 == 0:
            blur_amount += 1  # 確保為奇數

        # 應用高斯模糊
        blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), sigmaX=0)

        # 計算當前階段的權重圖
        weight = ((1 - focus_map_3ch) ** 2) * ((i + 1) / steps)  # 使用平方增強模糊效果

        # 累積模糊效果
        result = result * (1 - weight) + blurred_image * weight

    # 確保結果在 [0,1] 範圍內
    result = np.clip(result, 0, 1)

    # 轉換回 uint8 類型
    result = (result * 255).astype(np.uint8)

    return result

def stack_images(images, labels, scale=0.5):
    """
    將多張圖片並列顯示，並加上標籤。

    :param images: 圖片列表
    :param labels: 對應圖片的標籤列表
    :param scale: 縮放比例
    :return: 並列後的圖片
    """
    # 確保圖片大小一致
    resized_images = []
    for img in images:
        if len(img.shape) == 2:  # 灰度圖只有高度和寬度
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 調整大小
        height, width = img.shape[:2]
        img_resized = cv2.resize(img, (int(width * scale), int(height * scale)))
        resized_images.append(img_resized)

    # 將圖片水平串接
    stacked_image = np.hstack(resized_images)

    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # 白色文字

    for i, label in enumerate(labels):
        height, width = resized_images[i].shape[:2]
        cv2.putText(stacked_image, label, (i * width + 10, 20), font, font_scale, color, thickness)

    return stacked_image

def main():
    # 創建輸出資料夾
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 載入輸入影像
    input_image_path = 'DSC01197.JPG'  # 請替換為您的影像路徑
    image = cv2.imread(input_image_path)

    if image is None:
        print("錯誤：無法載入影像。")
        return

    # 將影像轉換為灰度圖
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "gray_image.jpg"), gray_image)

    # 應用高斯模糊降低雜訊
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_folder, "blurred_image.jpg"), blurred_image)

    # 使用 Canny 邊緣檢測，調整閾值
    edges = cv2.Canny(blurred_image, 100, 200)
    cv2.imwrite(os.path.join(output_folder, "edges.jpg"), edges)

    # 生成前景掩膜
    min_area = int(edges.shape[0] * edges.shape[1] * 0.01)
    foreground_mask = compute_foreground_mask(edges, min_area)
    cv2.imwrite(os.path.join(output_folder, "foreground_mask.jpg"), foreground_mask * 255)

    # 生成焦點權重圖
    focus_map = compute_focus_map_from_mask(foreground_mask)
    cv2.imwrite(os.path.join(output_folder, "focus_map.jpg"), (focus_map * 255).astype(np.uint8))

    # 估計原始圖像的焦外模糊程度
    current_blur = estimate_blur_amount(image, focus_map)
    print(f"當前焦外模糊程度：{current_blur:.4f}")

    # 指定期望的焦外模糊程度
    desired_blur = float(input("請輸入期望的焦外模糊程度(數值越大 模糊越強 例如2.0):"))

    # 計算需要增加的模糊量比例
    additional_blur_ratio = calculate_additional_blur(current_blur, desired_blur)
    print(f"需要增加的模糊量比例：{additional_blur_ratio:.2f}")

    # 根據增加的模糊量調整參數
    max_blur = int(100 * additional_blur_ratio)
    steps = int(5 * additional_blur_ratio)
    max_blur = np.clip(max_blur, 15, 255)
    steps = np.clip(steps, 3, 10)
    print(f"調整後的模糊參數：max_blur={max_blur}, steps={steps}")

    # 應用強模糊到焦外區域
    output_image = apply_strong_blur(image, focus_map, max_blur=max_blur, steps=steps)
    cv2.imwrite(os.path.join(output_folder, "output_image.jpg"), output_image)

    # 組合圖片
    images = [
        image,
        output_image,
        blurred_image,
        edges,
        foreground_mask * 255,
        (focus_map * 255).astype(np.uint8)
    ]
    labels = ['原始影像', '增強模糊效果', '高斯模糊', 'Canny 邊緣檢測', '前景掩膜', '焦點權重圖']
    result = stack_images(images, labels, scale=0.5)
    # 保存結果
    cv2.imwrite(os.path.join(output_folder, "combined_image_enhanced_blur.jpg"), result)

    # 顯示結果（可選）
    cv2.imshow('結果', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
