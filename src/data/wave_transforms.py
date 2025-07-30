import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import random
import numpy as np
import cv2


class WaveAugmentation:
    """波浪图像特定的数据增强"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return self.apply_wave_distortion(img)
        return img
    
    def apply_wave_distortion(self, img):
        """应用波浪形变，模拟水面波动效果"""
        try:
            # 将PIL图像转换为numpy数组
            img_array = np.array(img)
            h, w, c = img_array.shape
            
            # 创建波浪变形网格
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            
            # 添加轻微的波浪变形
            amplitude = random.uniform(1, 3)
            frequency = random.uniform(0.01, 0.03)
            
            X_wave = X + amplitude * np.sin(2 * np.pi * frequency * Y)
            Y_wave = Y + amplitude * np.cos(2 * np.pi * frequency * X)
            
            # 限制坐标范围
            X_wave = np.clip(X_wave, 0, w-1).astype(np.float32)
            Y_wave = np.clip(Y_wave, 0, h-1).astype(np.float32)
            
            # 使用OpenCV的remap进行高效变形
            distorted = cv2.remap(img_array, X_wave, Y_wave, cv2.INTER_LINEAR)
            
            return Image.fromarray(distorted.astype(np.uint8))
        
        except Exception as e:
            # 如果变形失败，返回原图
            return img


class FoamEnhancement:
    """增强泡沫效果"""
    
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            try:
                # 转换为numpy数组
                img_array = np.array(img)
                
                # 转换到HSV空间
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # 增强高亮度区域（泡沫）
                high_brightness_mask = hsv[:, :, 2] > 180
                
                # 降低饱和度，增加亮度
                hsv[high_brightness_mask, 1] *= 0.8
                hsv[high_brightness_mask, 2] = np.minimum(hsv[high_brightness_mask, 2] * 1.1, 255)
                
                # 转换回RGB
                enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                return Image.fromarray(enhanced)
            
            except Exception as e:
                return img
        
        return img


class WaterBlur:
    """模拟水面运动模糊"""
    
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            try:
                # 随机选择模糊方向和强度
                angle = random.uniform(0, 360)
                length = random.randint(3, 8)
                
                # 创建运动模糊核
                kernel = self._get_motion_blur_kernel(length, angle)
                
                # 应用模糊
                img_array = np.array(img)
                blurred = cv2.filter2D(img_array, -1, kernel)
                
                return Image.fromarray(blurred)
            
            except Exception as e:
                return img
        
        return img
    
    def _get_motion_blur_kernel(self, length, angle):
        """生成运动模糊核"""
        kernel = np.zeros((length, length))
        kernel[int((length-1)/2), :] = np.ones(length)
        kernel = kernel / length
        
        # 旋转核
        center = (length // 2, length // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))
        
        return kernel


class SeaColorAugmentation:
    """海水颜色增强"""
    
    def __init__(self, p=0.4):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            try:
                img_array = np.array(img).astype(np.float32)
                
                # 增强蓝色和绿色通道（海水特征）
                blue_factor = random.uniform(0.9, 1.2)
                green_factor = random.uniform(0.8, 1.1)
                red_factor = random.uniform(0.7, 1.0)
                
                img_array[:, :, 0] *= red_factor  # R
                img_array[:, :, 1] *= green_factor  # G
                img_array[:, :, 2] *= blue_factor  # B
                
                # 限制像素值范围
                img_array = np.clip(img_array, 0, 255)
                
                return Image.fromarray(img_array.astype(np.uint8))
            
            except Exception as e:
                return img
        
        return img