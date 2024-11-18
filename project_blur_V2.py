import cv2
import numpy as np

def estimate_depth_map(image):
    """
    估計景深圖的函數。
    """
    # 將影像轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用 Sobel 運算子計算梯度強度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 將梯度強度歸一化到 [0,1] 範圍
    depth_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # 反轉景深圖，使前景深度值高
    depth_map = 1 - depth_map
    
    # 對景深圖進行高斯模糊，平滑處理
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
    
    # 將景深圖轉換為 float32 類型
    depth_map = depth_map.astype(np.float32)
    
    return depth_map

def calculate_edge_density(image, window_size=15):
    """
    計算影像的邊緣密集度圖。

    :param image: 輸入影像（灰度圖）
    :param window_size: 用於計算邊緣密集度的窗口大小
    :return: 邊緣密集度圖，值範圍在 [0,1]
    """
    # 使用 Canny 邊緣檢測
    edges = cv2.Canny(image, 100, 200)
    
    # 創建全為 1 的卷積核，用於計算局部區域的邊緣總數
    kernel = np.ones((window_size, window_size), np.uint8)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
    
    # 將邊緣密集度圖歸一化到 [0,1] 範圍
    edge_density = cv2.normalize(edge_density, None, 0, 1, cv2.NORM_MINMAX)
    
    # 反轉邊緣密集度圖，讓邊緣密集度低的區域值較高
    edge_density = 1 - edge_density
    
    # 將邊緣密集度圖轉換為 float32 類型
    edge_density = edge_density.astype(np.float32)
    
    return edge_density

def combine_maps(depth_map, edge_density_map, alpha=0.7):
    """
    結合景深圖和邊緣密集度圖。

    :param depth_map: 景深圖，值範圍在 [0,1]
    :param edge_density_map: 邊緣密集度圖，值範圍在 [0,1]
    :param alpha: 權重係數，控制兩個圖的影響比例
    :return: 結合後的模糊權重圖
    """
    # 按權重結合兩個圖
    combined_map = cv2.addWeighted(depth_map, alpha, edge_density_map, 1 - alpha, 0)
    combined_map = cv2.normalize(combined_map, None, 0, 1, cv2.NORM_MINMAX)
    return combined_map

def apply_variable_blur(image, blur_weight_map, max_blur=80, depth_levels=20, exp_factor=1.5):
    """
    根據模糊權重圖對影像應用可變的高斯模糊，模糊半徑按照指數曲線遞增。

    :param image: 輸入影像
    :param blur_weight_map: 模糊權重圖，值範圍在 [0,1]
    :param max_blur: 最大模糊半徑
    :param depth_levels: 模糊級別數
    :param exp_factor: 指數基數（大於 1）
    :return: 應用模糊後的影像
    """
    # 將模糊權重圖量化為若干級
    blur_weight_quantized = (blur_weight_map * (depth_levels - 1)).astype(np.uint8)
    
    # 初始化輸出影像為原始影像
    output_image = image.copy()
    
    # 預先計算模糊半徑列表
    max_exp = exp_factor ** (depth_levels - 1)
    blur_amounts = []
    for i in range(depth_levels):
        # 計算指數增長的模糊半徑
        exp_value = exp_factor ** i
        blur_amount = int(max_blur * (exp_value - 1) / (max_exp - 1))
        if blur_amount % 2 == 0:
            blur_amount += 1  # 保證核大小為奇數
        blur_amounts.append(blur_amount)
        print(f"模糊級別：{i}, 模糊半徑：{blur_amount}")
    
    for i in range(depth_levels):
        blur_amount = blur_amounts[i]
        
        # 創建當前模糊級別的遮罩
        mask = (blur_weight_quantized == i).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            continue  # 如果沒有像素屬於該級別，則跳過
        
        # 應用高斯模糊
        if blur_amount > 1:
            blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
        else:
            blurred = image.copy()
        
        # 將模糊區域覆蓋到輸出影像
        mask_inv = cv2.bitwise_not(mask)
        # 保留輸出影像中非當前遮罩的區域
        output_image = cv2.bitwise_and(output_image, output_image, mask=mask_inv)
        # 將模糊後的區域添加到輸出影像
        temp = cv2.bitwise_and(blurred, blurred, mask=mask)
        output_image = cv2.add(output_image, temp)
    
    return output_image

def main():
    # 載入輸入影像
    input_image_path = 'DSCF5614.jpg'  # 請替換為您的影像路徑
    image = cv2.imread(input_image_path)
    
    if image is None:
        print("錯誤：無法載入影像。")
        return
    
    # 轉換為灰度圖，用於邊緣密集度計算
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 估計景深圖
    depth_map = estimate_depth_map(image)
    
    # 計算邊緣密集度圖
    edge_density_map = calculate_edge_density(gray_image, window_size=15)
    
    # 檢查資料類型和形狀
    print(f"depth_map.dtype: {depth_map.dtype}")
    print(f"edge_density_map.dtype: {edge_density_map.dtype}")
    print(f"depth_map.shape: {depth_map.shape}")
    print(f"edge_density_map.shape: {edge_density_map.shape}")
    
    # 結合景深圖和邊緣密集度圖
    combined_map = combine_maps(depth_map, edge_density_map, alpha=0.7)
    
    # 將模糊權重圖保存為圖像（可選）
    combined_map_visual = (combined_map * 255).astype(np.uint8)
    cv2.imwrite('combined_map.jpg', combined_map_visual)
    
    # 應用可變模糊，模擬更淺的景深
    output_image = apply_variable_blur(image, combined_map, max_blur=80, depth_levels=20, exp_factor=1.5)
    
    # 儲存輸出影像
    output_image_path = 'output.jpg'  # 請替換為您想要的輸出路徑
    cv2.imwrite(output_image_path, output_image)
    
    # 選擇性地顯示影像
    cv2.imshow('原始影像', image)
    cv2.imshow('景深圖', depth_map)
    cv2.imshow('邊緣密集度圖', edge_density_map)
    cv2.imshow('模糊權重圖', combined_map)
    cv2.imshow('模擬淺景深效果', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
