import cv2
import numpy as np

def estimate_depth_map(image):
    """
    改進的景深圖估計方法。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用梯度強度計算
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 歸一化
    depth_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # 反轉景深圖，使前景深度值高
    depth_map = 1 - depth_map
    
    # 平滑處理
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
    
    return depth_map

def apply_variable_blur(image, depth_map, max_blur=51, depth_levels=20, exp_factor=1.5):
    """
    根據景深圖對影像應用可變的高斯模糊，模糊半徑按照指數曲線遞增。
    
    :param image: 輸入影像
    :param depth_map: 景深圖
    :param max_blur: 最大模糊半徑
    :param depth_levels: 景深級別數
    :param exp_factor: 指數基數（大於 1）
    :return: 應用模糊後的影像
    """
    # 將景深圖歸一化到 [0, 1] 範圍
    depth_map_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    
    # 將景深圖量化為若干級，以提高處理效率
    depth_map_quantized = (depth_map_norm * (depth_levels - 1)).astype(np.uint8)
    
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
        print(f"景深級別：{i}, 模糊半徑：{blur_amount}")
    
    for i in range(depth_levels):
        blur_amount = blur_amounts[i]
        
        # 創建當前景深級別的遮罩
        mask = (depth_map_quantized == i).astype(np.uint8) * 255
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
    
    # 估計景深圖
    depth_map = estimate_depth_map(image)
    
    # 將景深圖保存為圖像
    depth_map_visual = (depth_map * 255).astype(np.uint8)
    cv2.imwrite('depth_map.jpg', depth_map_visual)
    
    # 應用可變模糊，模擬更淺的景深
    output_image = apply_variable_blur(image, depth_map, max_blur=80, depth_levels=20, exp_factor=1.5)
    
    # 儲存輸出影像
    output_image_path = 'output.jpg'  # 請替換為您想要的輸出路徑
    cv2.imwrite(output_image_path, output_image)
    
    # 選擇性地顯示影像
    cv2.imshow('原始影像', image)
    cv2.imshow('估計的景深圖', depth_map)
    cv2.imshow('模擬淺景深效果', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
