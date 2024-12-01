import cv2
import numpy as np

def estimate_depth_map(image):
    """
    估计景深图的函数。
    """
    # 将影像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用 Sobel 算子计算梯度强度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # 将梯度强度归一化到 [0,1] 范围
    depth_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # 反转景深图，使前景深度值高
    depth_map = 1 - depth_map
    
    # 对景深图进行高斯模糊，平滑处理
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
    
    # 将景深图转换为 float32 类型
    depth_map = depth_map.astype(np.float32)
    
    return depth_map

def calculate_edge_density(image, window_size=15):
    """
    计算影像的边缘密集度图。

    :param image: 输入影像（灰度图）
    :param window_size: 用于计算边缘密集度的窗口大小
    :return: 边缘密集度图，值范围在 [0,1]
    """
    # 使用 Canny 边缘检测
    edges = cv2.Canny(image, 100, 200)
    
    # 创建全为 1 的卷积核，用于计算局部区域的边缘总数
    kernel = np.ones((window_size, window_size), np.uint8)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
    
    # 将边缘密集度图归一化到 [0,1] 范围
    edge_density = cv2.normalize(edge_density, None, 0, 1, cv2.NORM_MINMAX)
    
    # 反转边缘密集度图，让边缘密集度低的区域值较高
    edge_density = 1 - edge_density
    
    # 将边缘密集度图转换为 float32 类型
    edge_density = edge_density.astype(np.float32)
    
    return edge_density

def combine_maps(depth_map, edge_density_map, alpha=0.7):
    """
    结合景深图和边缘密集度图。

    :param depth_map: 景深图，值范围在 [0,1]
    :param edge_density_map: 边缘密集度图，值范围在 [0,1]
    :param alpha: 权重系数，控制两个图的影响比例
    :return: 结合后的模糊权重图
    """
    # 按权重结合两个图
    combined_map = cv2.addWeighted(depth_map, alpha, edge_density_map, 1 - alpha, 0)
    combined_map = cv2.normalize(combined_map, None, 0, 1, cv2.NORM_MINMAX)
    return combined_map

def apply_variable_blur(image, blur_weight_map, max_blur=80, depth_levels=20, exp_factor=1.5):
    """
    根据模糊权重图对影像应用可变的高斯模糊，模糊半径按照指数曲线递增。

    :param image: 输入影像
    :param blur_weight_map: 模糊权重图，值范围在 [0,1]
    :param max_blur: 最大模糊半径
    :param depth_levels: 模糊级别数
    :param exp_factor: 指数基数（大于 1）
    :return: 应用模糊后的影像
    """
    # 将模糊权重图量化为若干级
    blur_weight_quantized = (blur_weight_map * (depth_levels - 1)).astype(np.uint8)
    
    # 初始化输出影像为原始影像
    output_image = image.copy()
    
    # 预先计算模糊半径列表
    max_exp = exp_factor ** (depth_levels - 1)
    blur_amounts = []
    for i in range(depth_levels):
        # 计算指数增长的模糊半径
        exp_value = exp_factor ** i
        blur_amount = int(max_blur * (exp_value - 1) / (max_exp - 1))
        if blur_amount % 2 == 0:
            blur_amount += 1  # 保证核大小为奇数
        blur_amounts.append(blur_amount)
        print(f"模糊级别：{i}, 模糊半径：{blur_amount}")
    
    for i in range(depth_levels):
        blur_amount = blur_amounts[i]
        
        # 创建当前模糊级别的遮罩
        mask = (blur_weight_quantized == i).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            continue  # 如果没有像素属于该级别，则跳过
        
        # 应用高斯模糊
        if blur_amount > 1:
            blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
        else:
            blurred = image.copy()
        
        # 将模糊区域覆盖到输出影像
        mask_inv = cv2.bitwise_not(mask)
        # 保留输出影像中非当前遮罩的区域
        output_image = cv2.bitwise_and(output_image, output_image, mask=mask_inv)
        # 将模糊后的区域添加到输出影像
        temp = cv2.bitwise_and(blurred, blurred, mask=mask)
        output_image = cv2.add(output_image, temp)
    
    return output_image

def main():
    # 载入输入影像
    input_image_path = 'DSCF5614.jpg'  # 请替换为您的影像路径
    image = cv2.imread(input_image_path)
    
    if image is None:
        print("错误：无法载入影像。")
        return
    
    # 转换为灰度图，用于边缘密集度计算
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 估计景深图
    depth_map = estimate_depth_map(image)
    
    # 计算边缘密集度图
    edge_density_map = calculate_edge_density(gray_image, window_size=15)
    
    # 检查数据类型和形状
    print(f"depth_map.dtype: {depth_map.dtype}")
    print(f"edge_density_map.dtype: {edge_density_map.dtype}")
    print(f"depth_map.shape: {depth_map.shape}")
    print(f"edge_density_map.shape: {edge_density_map.shape}")
    
    # 结合景深图和边缘密集度图
    combined_map = combine_maps(depth_map, edge_density_map, alpha=0.7)
    
    # 将景深图保存为图像
    depth_map_visual = (depth_map * 255).astype(np.uint8)
    cv2.imwrite('depth_map.jpg', depth_map_visual)
    
    # 将边缘密集度图保存为图像
    edge_density_map_visual = (edge_density_map * 255).astype(np.uint8)
    cv2.imwrite('edge_density_map.jpg', edge_density_map_visual)
    
    # 将模糊权重图保存为图像
    combined_map_visual = (combined_map * 255).astype(np.uint8)
    cv2.imwrite('combined_map.jpg', combined_map_visual)
    
    # 应用可变模糊，模拟更浅的景深
    output_image = apply_variable_blur(image, combined_map, max_blur=80, depth_levels=20, exp_factor=1.5)
    
    # 保存输出影像
    output_image_path = 'output.jpg'  # 请替换为您想要的输出路径
    cv2.imwrite(output_image_path, output_image)
    
    # 选择性地显示影像
    cv2.imshow('原始影像', image)
    cv2.imshow('景深图', depth_map_visual)
    cv2.imshow('边缘密集度图', edge_density_map_visual)
    cv2.imshow('模糊权重图', combined_map_visual)
    cv2.imshow('模拟浅景深效果', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
