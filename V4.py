import cv2
import numpy as np
import os

def compute_foreground_mask(edges, min_area):
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
    cv2.imwrite("binary_edges.jpg", binary_edges * 255)
    cv2.imwrite("dilated.jpg", dilated * 255)
    cv2.imwrite("closed.jpg", closed * 255)

    # 尋找所有連通區域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # 建立前景掩膜
    foreground_mask = np.zeros_like(labels, dtype=np.uint8)

    # 遍歷所有連通區域，跳過背景（標籤0）
    for i in range(1, num_labels):
        print(f"foregoundmask：{i}")
        area = stats[i, cv2.CC_STAT_AREA]

        # 根據面積和長寬比進行過濾
        if area >= min_area :
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

    # 確保資料型別為 float32
    image = image.astype(np.float32) / 255.0
    focus_map_3ch = focus_map_3ch.astype(np.float32)

    # 初始化結果影像
    result = image.copy()

    # 分階段累積模糊效果
    print(f"steps：{steps}")
    for i in range(steps):
        print(f"blur：{i}")
        # 計算當前階段的模糊程度
        blur_amount = int(max_blur * (i + 1) / steps)
        if blur_amount % 2 == 0:
            blur_amount += 1  # 確保為奇數

        # 應用高斯模糊
        blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), sigmaX=0)

        # 計算當前階段的權重圖
        weight = (1 - focus_map_3ch) * ((i + 1) / steps)

        # 累積模糊效果
        result = result * (1 - weight) + blurred_image * weight

    # 確保結果在 [0,1] 範圍內
    result = np.clip(result, 0, 1)

    # 轉換回 uint8 型別
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

    # 添加標籤（可選）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # 白色文字

    for i, label in enumerate(labels):
        height, width = resized_images[i].shape[:2]
        cv2.putText(stacked_image, label, (i * width + 10, 20), font, font_scale, color, thickness)

    return stacked_image

def main():
    # 載入輸入影像
    input_image_path = 'DSC01197.JPG'  # 請替換為您的影像路徑
    image = cv2.imread(input_image_path)

    if image is None:
        print("錯誤：無法載入影像。")
        return


    # 將影像轉換為灰度圖
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 應用高斯模糊降低雜訊
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 使用 Canny 邊緣檢測，調整閾值
    edges = cv2.Canny(blurred_image, 100, 200)

    # 生成前景掩膜
    min_area = int(edges.shape[0] * edges.shape[1] * 0.01)
    foreground_mask = compute_foreground_mask(edges, min_area)

    # 生成焦點權重圖
    focus_map = compute_focus_map_from_mask(foreground_mask)

    # 應用強模糊到焦外區域
    output_image = apply_strong_blur(image, focus_map, max_blur=100, steps=5)


    # 組合圖片
    images = [
        image,
        output_image,
        blurred_image,
        edges,
        foreground_mask * 255,
        (focus_map * 255).astype(np.uint8)
    ]
    labels = ['原始影像', '模擬淺景深效果','高斯模糊', 'Canny 邊緣檢測', '前景掩膜', '焦點權重圖']
    result = stack_images(images, labels, scale=0.5)
    # 顯示結果
    cv2.imwrite("combined_image_V4_C.jpg", result) 

if __name__ == '__main__':
    main()