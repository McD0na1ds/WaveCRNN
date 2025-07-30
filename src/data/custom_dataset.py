import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Optional, Callable, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class CustomImageDataset(Dataset):
    """
    自定义图像分类数据集
    支持多种数据组织方式，包括嵌套文件夹结构
    """

    def __init__(
            self,
            data_dir: str,
            mode: str = "train",
            transform: Optional[Callable] = None,
            data_format: str = "folder",
            annotation_file: Optional[str] = None,
            class_names: Optional[List[str]] = None,
            nested_folders: bool = True  # 新增参数，指示是否为嵌套文件夹结构
    ):
        """
        Args:
            data_dir: 数据根目录
            mode: 数据集模式 (train/val/test)
            transform: 图像变换
            data_format: 数据组织格式 (folder/csv/json)
            annotation_file: 标注文件路径
            class_names: 类别名称列表
            nested_folders: 是否为嵌套文件夹结构 (类别/样本文件夹/图像文件)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.data_format = data_format
        self.nested_folders = nested_folders

        # 加载数据
        if data_format == "folder":
            self.samples, self.class_to_idx = self._load_folder_data()
        elif data_format == "csv":
            self.samples, self.class_to_idx = self._load_csv_data(annotation_file)
        elif data_format == "json":
            self.samples, self.class_to_idx = self._load_json_data(annotation_file)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        # 设置类别名称
        if class_names is not None:
            self.classes = class_names
            # 重新映射class_to_idx以确保顺序一致
            old_class_to_idx = self.class_to_idx.copy()
            self.class_to_idx = {name: i for i, name in enumerate(class_names)}

            # 重新映射样本标签
            updated_samples = []
            for path, old_label in self.samples:
                # 找到旧标签对应的类别名称
                old_class_name = None
                for name, idx in old_class_to_idx.items():
                    if idx == old_label:
                        old_class_name = name
                        break

                if old_class_name in class_names:
                    new_label = self.class_to_idx[old_class_name]
                    updated_samples.append((path, new_label))
                else:
                    logger.warning(f"Class {old_class_name} not found in provided class_names")

            self.samples = updated_samples
        else:
            self.classes = list(self.class_to_idx.keys())

        self.num_classes = len(self.classes)

        logger.info(f"Loaded {len(self.samples)} samples for {mode} set")
        logger.info(f"Classes: {self.classes}")

    def _load_folder_data(self) -> Tuple[List[Tuple[str, int]], dict]:
        """从文件夹结构加载数据，支持嵌套结构"""
        mode_dir = os.path.join(self.data_dir, self.mode)
        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"Directory {mode_dir} not found")

        samples = []
        class_names = sorted([d for d in os.listdir(mode_dir)
                              if os.path.isdir(os.path.join(mode_dir, d))])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        logger.info(f"Found classes: {class_names}")

        for class_name in class_names:
            class_dir = os.path.join(mode_dir, class_name)
            class_idx = class_to_idx[class_name]

            if self.nested_folders:
                # 嵌套文件夹结构：类别/样本文件夹/图像文件
                sample_folders = [d for d in os.listdir(class_dir)
                                  if os.path.isdir(os.path.join(class_dir, d))]

                logger.info(f"Class {class_name}: found {len(sample_folders)} sample folders")

                for sample_folder in sample_folders:
                    sample_dir = os.path.join(class_dir, sample_folder)

                    # 在样本文件夹中查找图像文件
                    for item_name in os.listdir(sample_dir):
                        item_path = os.path.join(sample_dir, item_name)

                        if os.path.isfile(item_path) and self._is_image_file(item_path):
                            samples.append((item_path, class_idx))
            else:
                # 传统结构：类别/图像文件
                for item_name in os.listdir(class_dir):
                    item_path = os.path.join(class_dir, item_name)

                    if os.path.isfile(item_path) and self._is_image_file(item_path):
                        samples.append((item_path, class_idx))

        logger.info(f"Total samples found: {len(samples)}")
        return samples, class_to_idx

    def _is_image_file(self, file_path: str) -> bool:
        """检查文件是否为有效的图像文件"""
        # 首先检查扩展名
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')
        if file_path.lower().endswith(valid_extensions):
            return True

        # 如果没有扩展名，尝试打开文件验证是否为图像
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def _load_csv_data(self, csv_file: str) -> Tuple[List[Tuple[str, int]], dict]:
        """从CSV文件加载数据"""
        if csv_file is None:
            raise ValueError("CSV file path must be provided for CSV format")

        df = pd.read_csv(csv_file)

        # 过滤当前模式的数据
        if 'split' in df.columns:
            df = df[df['split'] == self.mode]

        samples = []
        unique_labels = sorted(df['label'].unique())
        class_to_idx = {label: i for i, label in enumerate(unique_labels)}

        for _, row in df.iterrows():
            img_path = os.path.join(self.data_dir, row['image_path'])
            label_idx = class_to_idx[row['label']]
            samples.append((img_path, label_idx))

        return samples, class_to_idx

    def _load_json_data(self, json_file: str) -> Tuple[List[Tuple[str, int]], dict]:
        """从JSON文件加载数据"""
        if json_file is None:
            raise ValueError("JSON file path must be provided for JSON format")

        with open(json_file, 'r') as f:
            data = json.load(f)

        samples = []

        # 获取类别信息
        if 'classes' in data:
            class_to_idx = {name: i for i, name in enumerate(data['classes'])}
        else:
            unique_labels = sorted(set(img['label'] for img in data['images']))
            class_to_idx = {label: i for i, label in enumerate(unique_labels)}

        # 过滤当前模式的数据
        for img_info in data['images']:
            if img_info.get('split', 'train') == self.mode:
                img_path = os.path.join(self.data_dir, img_info['path'])
                label_idx = class_to_idx[img_info['label']]
                samples.append((img_path, label_idx))

        return samples, class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为备选
            image = Image.new('RGB', (224, 224), color='white')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """计算类别权重，用于处理不平衡数据集"""
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1

        total_samples = len(self.samples)
        weights = []

        for i in range(self.num_classes):
            if i in label_counts:
                weight = total_samples / (self.num_classes * label_counts[i])
            else:
                weight = 1.0
            weights.append(weight)

        logger.info(f"Class distribution: {label_counts}")
        logger.info(f"Class weights: {weights}")

        return torch.FloatTensor(weights)

    def get_sample_info(self):
        """获取数据集样本信息，用于调试"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            'total_samples': len(self.samples),
            'num_classes': self.num_classes,
            'class_distribution': class_counts,
            'classes': self.classes
        }